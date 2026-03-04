#!/usr/bin/env python
# coding: utf-8

# # Parameters
# 
# 1. in_dim: int
#    - ノードの入力特徴量の数
# 2. hidden_dim: list of int
#    - **hidden_fefats[i]** はi番目のGAT layerから得られるノード表現のサイズ，デフォルトは **[64,64]**
#    - **len(hidden_dim)** はgraphSAGE layerの数
# 3. activation: list of activation functions or None
#    - **activation[i]** はi番目に適用する活性化関数，デフォルトは **ELU**
#    - **None** の場合、活性化関数を使用しない
# 4. dropout: list of float or None
#    - **dropout[i]** はi番目のGraphSAGE layerのdropout率を決定，デフォルトは **0.2**
#    - **len(dropout)** はGraphSAGE layerの数
# 5. aggregator_type : list of str
#    - **aggregator_type[i]** はi番目のGraphSAGE layerの集約方法を決定
#    - [mean, gcn, pool, lstm]から選択，デフォルトは **mean**

# In[1]:

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import SAGEConv
from dgl.nn.pytorch.glob import AvgPooling

__all__ = ["GraphSAGE"]

class GraphSAGE(nn.Module):
    def __init__(self,
                 config,
                 in_dim,
                 bias=None,
                 norm=None,
                 aggregator_type=None,
                ):
        super(GraphSAGE, self).__init__()

        num_block = config.MODEL.GNN.NUM_BLOCK

        in_dim = config.MODEL.GNN.EMBED_DIM

        dim = config.MODEL.GNN.EMBED_DIM
        hidden_dim=[dim,dim,dim]
        
        n_layers = len(hidden_dim)

        if bias is None:
            bias = [True for _ in range(n_layers)]

        self.layer_norm = nn.LayerNorm(hidden_dim[0])

        if norm is None:
            norm = [self.layer_norm for _ in range(n_layers)]

        if "leaky" in config.MODEL.GNN.ACTIVE:
            activations = [F.leaky_relu for _ in range(n_layers)]
        elif "elu" in config.MODEL.GNN.ACTIVE:
            activations = [F.elu for _ in range(n_layers)]
        activations.append(None)

        drop_rate = config.MODEL.GNN.DROP_RATE
        dropout = [drop_rate for _ in range(n_layers)]

        if aggregator_type is None:
            aggregator_type = ['mean' for _ in range(n_layers)]
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")        
        self.sage_blocks = nn.ModuleList()

        for block_idx in range(num_block):
            layers = nn.ModuleList()
            for layer_idx in range(n_layers):
                in_channel = in_dim if layer_idx==0 and block_idx==0 else hidden_dim[0]
                
                layers.append(
                    SAGEConv(
                        in_channel, 
                        hidden_dim[layer_idx], 
                        aggregator_type[layer_idx], 
                        dropout[layer_idx], 
                        bias[layer_idx],
                        norm[layer_idx],
                        activations[layer_idx]
                    )
                )
            self.sage_blocks.append(layers)

        self.pool = AvgPooling()
        
    def reset_parameters(self):
        for layers in self.sage_blocks:
            for sage in layers:
                sage.reset_parameters()
           
    def forward(self, g, feats, training):
        identity = feats
        
        for block_idx, layers in enumerate(self.sage_blocks):
            residual = feats
            for layer_idx, sage in enumerate(layers):
                feats = sage(g, feats)
                if  block_idx==len(self.sage_blocks)-1 and layer_idx==len(layers)-1:
                    feats = self.pool(g, feats)
                    return feats

            if residual.shape != feats.shape:
                linear = nn.Linear(residual.shape[1], feats.shape[1]).to(feats.device)
                residual = linear(residual)

            feats = feats + residual
            feats = self.layer_norm(feats)
           
        return feats
        

# In[ ]:





