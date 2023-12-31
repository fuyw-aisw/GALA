import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from pool import GraphMultisetTransformer
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import global_max_pool as gmp
import numpy as np
import math

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(input_dim, hidden_dim, bias=True), nn.ReLU(inplace=True),
                                 nn.Linear(hidden_dim, output_dim, bias=True), nn.ReLU(inplace=True))
        
    def forward(self, x):
        return self.mlp(x)
    
class GraphCNN(nn.Module):
    def __init__(self, channel_dims=[512, 512, 512], fc_dim=512, num_classes=256, pooling='MTP'):
        super(GraphCNN, self).__init__()

        # Define graph convolutional layers
        gcn_dims = [512] + channel_dims

        gcn_layers = [GCNConv(gcn_dims[i-1], gcn_dims[i], bias=True) for i in range(1, len(gcn_dims))]

        self.gcn = nn.ModuleList(gcn_layers)
        self.pooling = pooling
        if self.pooling == "MTP":
            self.pool = GraphMultisetTransformer(512, 256, 512, None, 10000, 0.25, ['GMPool_G', 'GMPool_G'], num_heads=8, layer_norm=True)
        else:
            self.pool = gmp
        # Define dropout
        self.drop1 = nn.Dropout(p=0.2) #0.2
    def activations_hook(self,grad): #
        self.final_conv_grads = grad #
    

    def forward(self, x, data, pertubed=False):
        #x = data.x
        # Compute graph convolutional part
        x = self.drop1(x)
        for idx, gcn_layer in enumerate(self.gcn):
            if idx == 0:
                x = F.relu(gcn_layer(x, data.edge_index.long()))
            elif idx == 2: #
                with torch.enable_grad(): #
                    self.final_conv_acts = gcn_layer(x, data.edge_index.long()) #
                self.final_conv_acts.register_hook(self.activations_hook) #
                x = x + F.relu(self.final_conv_acts) #
            else:
                x = x + F.relu(gcn_layer(x, data.edge_index.long()))
            
            if pertubed:
                random_noise = torch.rand_like(x).to(x.device)
                x = x + torch.sign(x) * F.normalize(random_noise, dim=-1) * 0.1
        if self.pooling == 'MTP':
            # Apply GraphMultisetTransformer Pooling
            g_level_feat = self.pool(x, data.batch, data.edge_index.long())
        else:
            g_level_feat = self.pool(x, data.batch)

        n_level_feat = x


        return n_level_feat, g_level_feat


class CL_protNET(torch.nn.Module):
    def __init__(self, out_dim, esm_embed=True, pooling='MTP'):
        super(CL_protNET,self).__init__()
        self.esm_embed = esm_embed
        #self.pertub = pertub
        self.out_dim = out_dim
        self.one_hot_embed = nn.Embedding(21, 96)
        self.proj_aa = nn.Linear(96, 512) 
        self.pooling = pooling
        #self.label_embed = label_embed
        #self.proj_spot = nn.Linear(19, 512)
        if esm_embed:
            self.proj_esm = nn.Linear(1280, 512)
            self.gcn = GraphCNN(pooling=pooling)
        else:
            self.gcn = GraphCNN(pooling=pooling)
        self.label_em = MLP(self.out_dim,1024,512)
        #self.esm_g_proj = nn.Linear(1280, 512)
        self.readout = nn.Sequential(
                        nn.Linear(512, 1024),
                        nn.ReLU(),
                        nn.Dropout(0.2), #0.2
                        nn.Linear(1024, out_dim),
                        nn.Sigmoid()
        )
     
    def forward(self, data, y):

        x_aa = self.one_hot_embed(data.native_x.long())
        x_aa = self.proj_aa(x_aa)
        
        if self.esm_embed:
            x = data.x.float()
            x_esm = self.proj_esm(x)
            x = F.relu(x_aa + x_esm)
            
        else:
            x = F.relu(x_aa)

        gcn_n_feat1, gcn_g_feat1 = self.gcn(x, data)
        g_feat_label = self.label_em(y)
        y_pred = self.readout(gcn_g_feat1)
        return y_pred, gcn_g_feat1, g_feat_label
        #if self.pertub:
        #   gcn_n_feat2, gcn_g_feat2 = self.gcn(x, data, pertubed=True) 
            
            
        #    return y_pred,gcn_g_feat1, gcn_g_feat2,
        #else:
        #    y_pred = self.readout(gcn_g_feat1)

        #    return y_pred
            
def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1
class RandomLayer(nn.Module):
    def __init__(self, input_dim_list=[], output_dim=512):
        super(RandomLayer, self).__init__()
        self.input_num = len(input_dim_list)
        self.output_dim = output_dim
        self.random_matrix = [torch.randn(input_dim_list[i], output_dim) for i in range(self.input_num)]

    def forward(self, input_list):
        device = input_list[0].device
        return_list = [torch.mm(input_list[i], self.random_matrix[i].to(device)) for i in range(self.input_num)]
        return_tensor = return_list[0] / math.pow(float(self.output_dim), 1.0/len(return_list))
        for single in return_list[1:]:
            return_tensor = torch.mul(return_tensor, single)
        return return_tensor

    def cuda(self):
        super(RandomLayer, self).cuda()
        self.random_matrix = [val.cuda() for val in self.random_matrix]
        
class AdversarialNetwork(nn.Module):
    def __init__(self, in_feature, hidden_size=512):
        super(AdversarialNetwork, self).__init__()
        self.ad_layer1 = nn.Linear(in_feature, hidden_size)
        self.ad_layer2 = nn.Linear(hidden_size, hidden_size)
        self.ad_layer3 = nn.Linear(hidden_size, 1)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5) #0.5
        self.dropout2 = nn.Dropout(0.5) #0.5
        self.sigmoid = nn.Sigmoid()
        self.apply(init_weights)
        self.iter_num = 0
        self.alpha = 10
        self.low = 0.0
        self.high = 1.0
        self.max_iter = 10000.0
    def forward(self, x):
        if self.training:
            self.iter_num += 1
        #print(self.iter_num)
        coeff = calc_coeff(self.iter_num, self.high, self.low, self.alpha, self.max_iter)
        #print(self.iter_num,coeff)
        x = x * 1.0
        x.register_hook(grl_hook(coeff))
        x = self.ad_layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.ad_layer2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        y = self.ad_layer3(x)
        y = self.sigmoid(y)
        return y

    def output_num(self):
        return 1
    def get_parameters(self):
        return [{"params":self.parameters(), "lr_mult":10, 'decay_mult':2}]
