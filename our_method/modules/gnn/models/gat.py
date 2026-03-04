#!/usr/bin/env python
# coding: utf-8

# # parameters
# 1. in_dim: int
#    - 入力ノードの特徴量の数
# 2. hidden_dim: list of int
#    - **hidden_dim[i]** はi番目のGAT layerのアテンションヘッドの出力次元，デフォルトは **[32,32]**
#    - **len(hidden_dim)** はGAT layerの数
# 3. num_nodes: list of int
#    - **num_nodes[i]** はi番目のGAT layerのアテンションヘッドの数，デフォルトは **4**
#    - **len(num_nodes)** はGAT layerの数
# 4. feat_drops: list of float
#    - **feat_drops[i]** はi番目のGAT layerにおける入力特徴量のdropout率，デフォルトは全GAT layerで **0**
#    - **len(feat_drops)** はGAT layerの数
# 5. attn_drops: lis of float
#    - **attn_drops[i]** はi番目のGAT layerにおけるエッジのアテンション値のdropout率，デフォルトは **0**
#    - **len(attn_drops)** はGAT layerの数
# 6. alphas : list of float
#    - LeakyReLUのパラメータであり，入力値の定数倍に使用 (x < 0)
#    - **alpha[i]** はi番目のGAT layerにおけるslope，デフォルトは全layerで **0.2**
#    - **len(alpha)** はGAT layerの数
# 7. residuals : list of bool
#    - **residuals[i]** i番目のGAT layerで残差接続を使用するか決定
#    - **len(residuals)** はGAT layerの数
# 8. agg_modes : list of str
#    - 各GAT layerにおけるMulti Head Attentionの結果の集約方法
#    - **flatten** は全てのheadの結果を結合
#    - **mean** は全てのheadの結果を平均
#    - **agg_modes[i]** はi番目のGAT layerにおけるMHAの結果の集約方法，デフォルトは **flatten**
# 9. activations : list of activation function or None
#    - **activations[i]** はi番目のGAT layerにおけるMHAの結果の活性化関数，デフォルトではELU(Exponential Linear Unit)
#    - **len(activations)** はGAT layerの数
# 10. biases : list of bool
#     - **biases[i]** はi番目のGAT layerでバイアスを使用するか決定，デフォルトでは使用する
#     - **len(biases)** はi番目のGAT layerの数
# 11. allow_zero_in_degree: bool
#     - 全てのlayerで **次数が0** のノードを許可するか決定，デフォルトでは許可しない

# In[5]:


import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv
import torch
from dgl.nn.pytorch.glob import AvgPooling

__all__ = ["GAT"]

class GAT(nn.Module):
    def __init__(
        self,
        config,
        in_dim=None,
        device="cpu"
    ):
        super(GAT,self).__init__()
        # embedding dim
        dim = config.MODEL.GNN.EMBED_DIM
        hidden_dim = [dim,dim,dim]
        n_layers = len(hidden_dim)

        # head num in multi head attention 
        head_num = config.MODEL.GNN.HEAD_NUM
        num_heads = [head_num if i<n_layers else 1 for i in range(n_layers)] 

        # dropout
        drop_rate = config.MODEL.GNN.DROP_RATE
        feat_drops = [drop_rate for _ in range(n_layers)]    
        attn_drops = [drop_rate for _ in range(n_layers)]

        alpha = config.MODEL.GNN.ALPHA
        alphas = [alpha for _ in range(n_layers)]

        # residual
        residuals = [True for _ in range(n_layers)]
        
        # activation
        if "leaky" in config.MODEL.GNN.ACTIVE:
            activations = [F.leaky_relu for _ in range(n_layers)]
        elif "elu" in config.MODEL.GNN.ACTIVE:
            activations = [F.elu for _ in range(n_layers)]

        num_block = config.MODEL.GNN.NUM_BLOCK
        self.is_fast = config.MODEL.GNN.SAMPLE_MODEL == "fast"
        
        lengths = [
            len(hidden_dim),
            len(num_heads),
            len(feat_drops),
            len(attn_drops),
            len(alphas),
            len(residuals),
            len(activations),
        ]

        assert len(set(lengths)) == 1,(            
            "Expect the lengths of hidden_dim, num_heads, "
            "feat_drops, attn_drops, alphas, residuals, "
            "activations to be the same, "
            "got {}".format(lengths)
        )

        self.in_dim = in_dim
        self.device = device
        self.gat_blocks = nn.ModuleList()

        for block_idx in range(num_block):
            gat_layers = nn.ModuleList()
            for layer_idx in range(n_layers):
                in_channel = in_dim if block_idx==0 and layer_idx==0 else hidden_dim[0]
                gat_layers.append(
                    GATConv(
                        in_channel,
                        hidden_dim[layer_idx],
                        num_heads[layer_idx],
                        feat_drops[layer_idx],
                        attn_drops[layer_idx],
                        alphas[layer_idx],
                        residuals[layer_idx],
                        activations[layer_idx],
                        allow_zero_in_degree=config.MODEL.GNN.ALLOW_ZERO
                    )
                )
            self.gat_blocks.append(gat_layers)
            
        self.layer_norm = nn.LayerNorm(hidden_dim[0])
        self.pool = AvgPooling()

    def reset_parameters(self):
        for layers in self.gat_blocks:
            for gat in layers:
                gat.reset_parameters()

    def forward(self, g, feats, training):
        for block_idx, layers in enumerate(self.gat_blocks):
            for layer_idx, gnn in enumerate(layers):
                feats = gnn(g, feats) 
                feats = torch.mean(feats, dim=1)
                feats = self.layer_norm(feats)

        s = self.pool(g, feats)
        return feats, s

# In[ ]:



