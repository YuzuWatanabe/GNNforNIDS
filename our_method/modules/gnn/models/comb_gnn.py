#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import torch
from torch import nn
import torch.nn.functional as F

class CombinedGNN(nn.Module):
    def __init__(self, config, gat, fast, device):
        super(CombinedGNN, self).__init__()
        self.device = device
        self.gat = gat.to(device)
        self.fast = fast.to(device)

        # GNNとFastのノード特徴量を並列的に結合
        comb_embed_dim = 64 + 128 
        self.comb_feat_linear = nn.Linear(comb_embed_dim, 128).to(self.device) 
        self.feat_norm = nn.LayerNorm(128).to(self.device)

        # GNNとFastのサマリベクトルを並列的に結合
        self.comb_s_linear = nn.Linear(comb_embed_dim, 128).to(self.device)
        self.s_norm = nn.LayerNorm(128).to(self.device)

    def forward(self, batched_graph, graph_feats, training):
        batched_graph = batched_graph.to(self.device)
        graph_feats = graph_feats.to(self.device)
        # gatの順伝播処理
        gat_feat, gat_s = self.gat(batched_graph, graph_feats, training)
        
        # fastの順伝搬処理
        fast_feat, fast_s = self.fast(batched_graph, graph_feats, training)

        # ノード特徴量の結合
        comb_feat = torch.cat([gat_feat.float(), fast_feat.float()], dim=1)
        hidden_feat = self.comb_feat_linear(comb_feat)
        hidden_feat = F.leaky_relu(hidden_feat)
        hidden_feat = self.feat_norm(hidden_feat)
        hidden_feat = F.dropout(hidden_feat, p=0.2, training=training)

        # グラフサマリベクトルの結合
        comb_s = torch.cat([gat_s.float(), fast_s.float()], dim=1)
        hidden_s = self.comb_feat_linear(comb_s)
        hidden_s = F.leaky_relu(hidden_s)
        hidden_s = self.s_norm(hidden_s)
        hidden_s = F.dropout(hidden_s, p=0.2, training=training)

        return hidden_feat.float(), hidden_s.float()

