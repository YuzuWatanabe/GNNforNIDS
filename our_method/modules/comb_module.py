#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import torch
from torch import nn
import torch.nn.functional as F

class CombinedModule(nn.Module):
    def __init__(self, config, gnn_module, swin_model, device):
        super(CombinedModule, self).__init__()
        self.device = device
        self.gnn_module = gnn_module.to(self.device)
        gnn_embed_dim = config.MODEL.GNN.FAST_EMBED_DIM

        self.swin_model = swin_model.to(self.device)
        swin_embed_dim = config.MODEL.SWIN.EMBED_DIM
            
        # LLM+GNNの結合層
        comb_embed_dim1 = gnn_embed_dim + swin_embed_dim
        self.comb_fc = nn.Linear(comb_embed_dim1, swin_embed_dim).to(self.device)
        self.layer_norm = nn.LayerNorm(swin_embed_dim).to(self.device)

        num_class = config.MODEL.NUM_CLASSES

        self.fc1 = nn.Linear(128, 64).to(self.device) # 128 -> 64
        self.fc2 = nn.Linear(64, num_class).to(self.device) # 64 -> 11

    def forward(self, batched_raw, batched_graph, training):
        graph_feats = batched_graph.ndata['feat'].float().to(self.device)
        
        # gnn moduleの順伝播処理
        _, hidden_s = self.gnn_module(batched_graph, graph_feats, training)
        
        # swin modekの順伝播処理
        swin_output = self.swin_model(batched_raw)
        swin_output.float()
        combined_output = torch.cat([swin_output, hidden_s], dim=1)           
        hidden = self.comb_fc(combined_output)
        hidden = F.elu(hidden)
        hidden = self.layer_norm(hidden)
        hidden = F.dropout(hidden, p=0.2, training=training)
        
        if hidden.size()[1] == 128:
            hidden = self.fc1(hidden)

        out = self.fc2(hidden)

        return out