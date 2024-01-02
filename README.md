# Learning A Generalized Graph Transformer for Protein Function Prediction in Dissimilar Sequences
---
## Overview
In this work, we propose a new generalized approach named Graph Adversarial Learning with Alignment (GALA) for protein function prediction. To encode all these proteins from different environments, our GALA utilizes a graph Transformer architecture with the attention mechanism for unified protein representations. More importantly, GALA introduces a domain discriminator conditioned on both representations and predicted labels, which is adversarially trained to ensure the invariance of representation across different environments. To make the best of label information, we generate label embeddings in the hidden space, which would be aligned explicitly with protein representations. We conduct extensive experiments on various benchmark datasets and our GALA can outperform all the compared baselines with high generalizability.
<img src="sortedmodel/frame-final.png">
## Installation
Start by flowwing this source codes:
```bash
git clone https://github.com/fuyw-aisw/GALA.git
cd GALA
conda create -n gala python=3.7
conda activate gala
step1: install PyTorch’s CUDA support on Linux
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
step2: install pyg package

step3: download other dependencies

```

