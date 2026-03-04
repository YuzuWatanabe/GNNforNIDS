#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
from our_method.modules.gnn.pretrain.losses.losses import Loss

# Barlow Twin損失の計算
def bt_loss(h1: torch.Tensor, h2: torch.Tensor, lambda_, batch_norm=True, eps=1e-15, *args, **kwargs):
    batch_size = h1.size(0)
    feature_dim = h1.size(1)

    if lambda_ is None:
        lambda_ = 1. / feature_dim

    std1 = h1.std(dim=0)
    std2 = h2.std(dim=0)

    if batch_norm:
        z1_norm = (h1 - h1.mean(dim=0)) / (std1 + eps)
        z2_norm = (h2 - h2.mean(dim=0)) / (std2 + eps)
        c = (z1_norm.T @ z2_norm) / batch_size
    else:
        # 相関行列の算出
        c = h1.T @ h2 / batch_size

    off_diagonal_mask = ~torch.eye(feature_dim).bool()
    loss = (1 - c.diagonal()).pow(2).sum()
    loss += lambda_ * c[off_diagonal_mask].pow(2).sum()

    return loss
    
class BarlowTwins(Loss):
    def __init__(self, lambda_: float = 5e-3, batch_norm: bool = True, eps: float = 1e-15):
        super(BarlowTwins, self).__init__()
        self.lambda_ = lambda_
        self.batch_norm = batch_norm
        self.eps = eps

    def compute(self, anchor, sample, pos_mask, neg_mask, *args, **kwargs) -> torch.FloatTensor:
        loss = bt_loss(anchor, sample, self.lambda_, self.batch_norm, self.eps)
        return loss

