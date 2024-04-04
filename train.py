from graph_data import GoTermDataset, collate_fn
from torch.utils.data import DataLoader
from network import CL_protNET, MLP
from nt_xent import NT_Xent
import nt_xent
import network
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import torch.nn as nn
from sklearn import metrics
from utils import log
import argparse
from config import get_config
import numpy as np
import random
import time
import warnings
warnings.filterwarnings("ignore")

def train(config, task, suffix):
    #train:val:test=5:2.5:2.5
    t1 = time.time()
    train_set = GoTermDataset("train", task, config.AF2model)
    #pos_weights = torch.tensor(train_set.pos_weights).float()
    valid_set = GoTermDataset("val", task, config.AF2model)
    test_set = GoTermDataset("test", task, config.AF2model)
    t2 = time.time()
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn,drop_last=True)
    val_loader = DataLoader(valid_set, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn,drop_last=True)
    t3 = time.time()
    print("set",t2-t1)
    print("loader",t3-t2)
    output_dim = valid_set.y_true.shape[-1] #number of class
    ## set basic network
    model = CL_protNET(output_dim, config.esmembed, config.pooling).to(config.device)
    optimizer = torch.optim.Adam(
        params = model.parameters(), 
        **config.optimizer,
        )
    # add additional network for some methods
    if args.random:
        random_layer = network.RandomLayer([512,output_dim],512).to(config.device)
        ad_net = network.AdversarialNetwork(512,512).to(config.device)
        #random_layer.cuda()
    else:
        random_layer = None
        ad_net = network.AdversarialNetwork(512*output_dim,512).to(config.device)
    #ad_net = ad_net.cuda()   
    #optimizer_ad = torch.optim.Adam(ad_net.parameters(), lr=1e-4)
    optimizer_ad = torch.optim.SGD(ad_net.parameters(), lr=0.03, weight_decay=0.0005, momentum=0.9)

    
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **config.scheduler)
    bce_loss = torch.nn.BCELoss(reduce=False)
    
    
    # categorical embedding                                                                           
    train_loss = []
    val_loss = []
    val_aupr = []
    val_Fmax = []
    es = 0
    y_true_all = valid_set.y_true.float().reshape(-1)
    
    #target_label = test_set.y_true.float().to(config.device)
    #target_bank = torch.randn(len(test_set),512).to(config.device)
    #y_non = torch.zeros(config.batch_size,output_dim).to(config.device)
    alpha = config.alpha
    gamma = config.gamma
    eta = 1
    pre_epoch = -1
    
    len_source = len(train_loader)
    len_target = len(test_loader)
    if len_source > len_target:
        num_iter = len_source
    else:
        num_iter = len_target
        
    for ith_epoch in range(config.max_epochs):
        model.train()
        ad_net.train()
        for idx_batch in range(num_iter):
            if idx_batch % len_source == 0:
                iter_source = iter(train_loader)
            if idx_batch % len_target == 0:
                iter_target = iter(test_loader)
            x_source, y_source = iter_source.next()
            x_target, _ = iter_target.next()
            x_source = x_source.to(config.device)
            y_source = y_source.to(config.device)
            x_target = x_source.to(config.device)
            optimizer.zero_grad()
            optimizer_ad.zero_grad()
            y_pred_source, g_feat_source, g_label_source = model(x_source,y_source)
            y_pred_target, g_feat_target = model(x_target) #target
            features = torch.cat((g_feat_source, g_feat_target), dim=0)
            outputs = torch.cat((y_pred_source, y_pred_target), dim=0)
            softmax_out = nn.Softmax(dim=1)(outputs)
            sup_loss = bce_loss(y_pred_source, y_source) #* pos_weights.to(config.device)
            sup_loss = sup_loss.mean()
            criterion = NT_Xent(g_feat_source.shape[0], 0.1, 1)
            source_loss = alpha * criterion(g_feat_source, g_label_source)
            #target_loss = beta * criterion(g_feat_target, g_feat_target1)
            
            if config['method'] == 'CDAN-E':
                entropy = nt_xent.Entropy(softmax_out)
                transfer_loss = gamma * nt_xent.CDAN([features, softmax_out], ad_net, entropy, network.calc_coeff(num_iter*ith_epoch+idx_batch), random_layer)
            elif config['method']  == 'CDAN':
                transfer_loss = gamma * nt_xent.CDAN([features, softmax_out], ad_net, None, None, random_layer)
            elif config['method']  == 'DANN':
                transfer_loss = gamma * nt_xent.DANN(features, ad_net)
            else:
                raise ValueError('Method cannot be recognized.')
                #loss = sup_loss + transfer_loss + source_loss + target_loss
            loss = sup_loss + transfer_loss + source_loss
            log(f"{idx_batch}/{ith_epoch} train_epoch ||| Loss: {round(float(loss),3)} ||| sup_loss: {round(float(sup_loss),3)}||| transfer_loss: {round(float(transfer_loss),4)} ||| source_loss: {round(float(source_loss),3)}")
                
            train_loss.append(loss.clone().detach().cpu().numpy())

            loss.backward()
            optimizer.step()
            optimizer_ad.step()
            
            
            
            #for name, param in ad_net.named_parameters():
            #    print(f"Parameter: {name}, Value: {param}")

        
        eval_loss = 0
        model.eval()
        y_pred_all = []
        n_nce_all = []
        
        with torch.no_grad():
            for idx_batch, batch in enumerate(val_loader):
                y_pred,_= model(batch[0].to(config.device))
                y_pred_all.append(y_pred)
            y_pred_all = torch.cat(y_pred_all, dim=0).cpu().reshape(-1)
            eval_loss = bce_loss(y_pred_all, y_true_all).mean()
                
            aupr = metrics.average_precision_score(y_true_all.numpy(), y_pred_all.numpy(), average="samples")
            val_aupr.append(aupr)
            log(f"{ith_epoch} VAL_epoch ||| loss: {round(float(eval_loss),3)} ||| aupr: {round(float(aupr),3)}")
            val_loss.append(eval_loss.numpy())
            if ith_epoch == 0:
                best_eval_loss = eval_loss
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                es = 0
            else:
                es += 1
                print("Counter {} of 5".format(es))

            if es > 4 :
                torch.save(model.state_dict(), config.model_save_path + task + f"{suffix}.pt")
                torch.save(
                    {
                        "train_bce": train_loss,
                        "val_bce": val_loss,
                        "val_aupr": val_aupr,
                    }, config.loss_save_path + task + f"{suffix}.pt"
                )

                break
                

def str2bool(v):
    if isinstance(v,bool):
        return v
    if v == 'True' or v == 'true':
        return True
    if v == 'False' or v == 'false':
        return False


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--task', type=str, default='bp', choices=['bp','mf','cc'], help='')
    p.add_argument('--suffix', type=str, default='', help='')
    p.add_argument('--device', type=str, default='', help='')
    p.add_argument('--esmembed', default=False, type=str2bool, help='')
    p.add_argument('--pooling', default='MTP', type=str, choices=['MTP','GMP'], help='Multi-set transformer pooling or Global max pooling')
    #p.add_argument('--contrast', default=True, type=str2bool, help='whether to do contrastive learning')
    p.add_argument('--AF2model', default=False, type=str2bool, help='whether to use AF2model for training')
    p.add_argument('--batch_size', type=int, default=32, help='')
    p.add_argument('--method', type=str, default='CDAN-E', choices=['CDAN', 'CDAN-E', 'DANN'])
    p.add_argument('--random', type=bool, default=True, help='whether to use random')
    p.add_argument('--alpha', type=float, default=0.05)
    p.add_argument('--gamma', type=float, default=1)
    args = p.parse_args()
    config = get_config()
    config.optimizer['lr'] = 1e-4
    config.batch_size = args.batch_size
    config.max_epochs = 100
    if args.device != '':
        config.device = "cuda:" + args.device
    config.esmembed = args.esmembed
    print(args)
    config.pooling = args.pooling
    #config.contrast = args.contrast
    config.AF2model = args.AF2model
    config.random = args.random
    config.method = args.method
    config.alpha = args.alpha
    config.gamma = args.gamma
    
    train(config, args.task, args.suffix)

