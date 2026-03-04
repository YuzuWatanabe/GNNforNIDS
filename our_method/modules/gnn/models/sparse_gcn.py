#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.sparse as dglsp

class SparseGCNLayer(nn.Module):
    def __init__(self, 
                 in_dim, 
                 hidden_dim, 
                 bias=True, 
                 activation=None, 
                 allow_zero_in_degree=False,
                 dropout_rate=0.0,
                 layer_norm=None,
                 device='cpu'
                ):
        super(SparseGCNLayer, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.allow_zero_in_degree = allow_zero_in_degree
        self.layer_norm = layer_norm
        self.dropout = nn.Dropout(dropout_rate)
        self.device = device
        self.negative_slope = 0.01

        self.W = nn.Linear(in_dim, hidden_dim, bias=bias).to(device)
    
    def forward(self, adj, feats):
        adj = adj.to(self.device)
        feats = feats.to(self.device)
        support = self.W(feats)
        hidden = torch.sparse.mm(adj, support)
        hidden = self.layer_norm(hidden)
        hidden = self.activation(hidden)
        output = self.dropout(hidden)

        return output

# In[ ]: