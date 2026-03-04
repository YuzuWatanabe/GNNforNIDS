#!/usr/bin/env python
# coding: utf-8

# In[4]:


import torch.nn as nn
import torch.nn.functional as F
import pdb
import math
import torch
import numpy as np
import scipy.sparse as sp

import dgl
from dgl.nn import GraphConv
from dgl.nn.pytorch.glob import AvgPooling
import dgl.sparse as dglsp

from scipy.sparse.linalg import norm as sparse_norm
from torch.nn.parameter import Parameter

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.sparse as dglsp

from .sparse_gcn import SparseGCNLayer

class FastGCN(nn.Module):
    def __init__(self, config, in_dim, device="cpu"):
        super(FastGCN, self).__init__()

        num_block = config.MODEL.GNN.FAST_NUM_BLOCK

        """
        # 直列接続の場合
        in_dim = config.MODEL.GNN.FAST_EMBED_DIM
        """
        
        # 並列接続の場合
        in_dim = in_dim

        # FastGCN Embedding dim 
        dim = config.MODEL.GNN.FAST_EMBED_DIM
        hidden_dim= [dim,dim]

        n_layers = len(hidden_dim)

        # bias
        bias = True

        # activation
        if "leaky" in config.MODEL.GNN.ACTIVE:
            activations = F.leaky_relu
        elif "elu" in config.MODEL.GNN.ACTIVE:
            activations = F.elu
        
        # normalization
        layer_norm = nn.LayerNorm(hidden_dim[0])
        self.fast_blocks = nn.ModuleList()
        self.device = device

        for block_idx in range(num_block):
            fast_layer = nn.ModuleList()
            for layer_idx in range(n_layers):
                in_channel = in_dim if layer_idx==0 and block_idx==0 else hidden_dim[0]

                fast_layer.append(
                    SparseGCNLayer(
                        in_channel, 
                        hidden_dim[layer_idx], 
                        bias,
                        activations,
                        config.MODEL.GNN.ALLOW_ZERO,
                        config.MODEL.GNN.DROP_RATE,
                        layer_norm,
                        self.device
                    )
                )
                
            self.fast_blocks.append(fast_layer)

        self.pool = AvgPooling()
        self.sampler = Sampler_FastGCN(config, in_dim, hidden_dim, self.device)
        self.layer_norm = nn.LayerNorm(hidden_dim[0])
        
    def reset_parameters(self):
        for layers in self.fast_blocks:
            for fast in layers:
                fast.reset_parameters()

    def forward(self, g, feats, training):
        residual = feats
        
        for block_idx, layers in enumerate(self.fast_blocks):
            sample_adj, feats = self.sampler.graph_sampling(g, feats) #層別サンプリング
            for layer_idx, fastgcn in enumerate(layers):
                feats = fastgcn(sample_adj[layer_idx], feats)

            if residual.shape != feats.shape:
                linear = nn.Linear(residual.shape[1], feats.shape[1]).to(self.device)
                residual = linear(residual)
            
            feats = feats + residual
            feats = self.layer_norm(feats)

        s = self.pool(g, feats)
        
        return feats, s

class Sampler:
    def __init__(self, in_dim, layer_size, device):
        self.in_dim = in_dim
        self.layer_sizes = layer_size
        self.device = device
        self.num_layers = len(self.layer_sizes)

    def graph_sampling(self, g, feats):
        raise NotImplementedError("sampling is not implimented")

class Sampler_FastGCN(Sampler):
    def __init__(self, config, in_dim, layer_size, device):
        super().__init__(in_dim, layer_size, device)
        self.probs = None
        self.num_sample = config.MODEL.GNN.NODE_SAMPLE_NUM

    def graph_sampling(self, g, feats):
        indices = torch.stack(g.edges())
        N = g.num_nodes()
        dgl_adj = dglsp.spmatrix(indices, shape=(N,N))

        # torch sparseで処理を実装
        feats = feats.to(self.device)
        norm_coo_adj = normalize_adj(dgl_adj, self.device)
    
        probs = torch.sparse.sum(norm_coo_adj, dim=0).to_dense()
        self.probs = probs / probs.sum()
        self.probs = self.probs.to(self.device)
        
        all_support = [[]] * self.num_layers # 各レイヤーの隣接行列
        
        cur_out_node = torch.arange(N)
        
        for layer_index in range(self.num_layers-1, -1, -1):
            # l -> l-1 -> l-2のように最終層から逆順に必要なノードをサンプリング
            # 階層サンプリング
            cur_sampled, cur_support = self._one_layer_sampling(norm_coo_adj, cur_out_node, self.num_sample[layer_index])
            all_support[layer_index] = cur_support
            cur_out_node = cur_sampled

        #all_support = self._change_sparse_to_tensor(all_support)
        sample_feats0 = feats[cur_out_node]
        return all_support, sample_feats0

    def _one_layer_sampling(self, norm_adj, output_nodes, sample_num):
        norm_adj = norm_adj.coalesce()
        indices = norm_adj.indices()
        values = norm_adj.values()
        N = norm_adj.shape[1]

        output_nodes = output_nodes.to(self.device)
        
        # 出力ノードの行に対応するエッジ抽出
        output_mask = torch.zeros(norm_adj.shape[0], dtype=torch.bool, device=self.device)
        output_mask.scatter_(0, output_nodes, True)
        
        row_mask = output_mask[indices[0]]
        sel_rows = indices[:, row_mask]
        sel_values = values[row_mask]

        # 隣接ノードの候補 (neis)
        col_indices = sel_rows[1]
        neis = torch.unique_consecutive(col_indices.sort()[0])

        # サンプリング確率
        p1 = self.probs[neis]
        p1 = torch.nan_to_num(p1, nan=0.0, posinf=0.0, neginf=0.0)
        p_sum = p1.sum()
        p1 = p1 / p_sum

        # サンプリング実行
        sampled_idx = self.weighted_reservoir_sampling(p1, sample_num, self.device)
        u_sampled = neis[sampled_idx]
        u_sampled, inverse_indices = torch.unique(u_sampled, sorted=True, return_inverse=True)
        sampled_p1 = p1[sampled_idx]

        # 高速マッピング用にソート済みにしておく
        u_sampled_sorted, sort_idx = torch.sort(u_sampled)
        sel_col = sel_rows[1]
        pos = torch.searchsorted(u_sampled_sorted, sel_col)

        # マッピングが sample_num 未満の位置のみ使用
        valid = pos < sample_num
        sel_rows = sel_rows[:, valid]
        sel_values = sel_values[valid]
        pos = pos[valid]

        scale = 1.0 / (sampled_p1[pos] * sample_num)
        scaled_values = sel_values * scale

        # スパーステンソル生成
        support_sparse = torch.sparse_coo_tensor(
            torch.stack([sel_rows[0], pos], dim=0),
            scaled_values,
            size=(len(output_nodes), len(u_sampled)),
            dtype=torch.float32,
            device=self.device
        )

        return u_sampled_sorted, support_sparse

    def weighted_reservoir_sampling(self, weights, k, device):
        device = self.device
        N = weights.shape[0]

        if k >= N:
            return torch.arange(N, device=device)

        # weights が 0 の場合は除外
        eps = 1e-12
        weights = weights.clamp(min=eps)

        u = torch.rand(N, device=device)
        gumbel = -torch.log(-torch.log(u + eps) + eps)
        scores = gumbel / weights

        # 最小スコアk個のインデックスを返す
        indices = torch.topk(-scores, k).indices  # 最大-k → 最小
        return indices

def normalize_adj(adj, device):
    adj = adj.coalesce()
    indices = adj.indices()
    values = adj.val.to(device)   # shape: [NNZ]
    N = adj.shape[0]
    crow, col = indices[0], indices[1]
    crow = crow.to(device)
    col = col.to(device)
    deg = torch.zeros(N, dtype=values.dtype, device=device)
    deg.scatter_add_(0, crow, values)
    d_inv_sqrt = deg.clamp(min=1e-12).rsqrt()
    new_values = d_inv_sqrt[crow] * values * d_inv_sqrt[col]
    
    return torch.sparse_coo_tensor(indices, new_values, adj.shape, device=device)

