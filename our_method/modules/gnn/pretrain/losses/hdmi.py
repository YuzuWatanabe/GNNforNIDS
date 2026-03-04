#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch 
import torch.nn as nn
from our_method.modules.gnn.pretrain.losses.losses import Loss

def hdmi_loss(score_pos:torch.Tensor, score_neg:torch.Tensor, criterion, eps:float):
    score_pos = torch.clamp(score_pos, eps, 1.0-eps)
    score_neg = torch.clamp(score_neg, eps, 1.0-eps)
    score_pos = torch.unsqueeze(score_pos, dim=0)
    score_neg = torch.unsqueeze(score_neg, dim=0)
    score = torch.cat([score_pos, score_neg], dim=1)
    
    pos_targets = torch.ones_like(score_pos, dtype=torch.float32)
    neg_targets = torch.zeros_like(score_neg, dtype=torch.float32)
    targets = torch.cat([pos_targets, neg_targets], dim=1)
    loss = criterion(score, targets) 
    
    #loss = -(pos_targets * torch.log(score_pos+eps)) + -((1-neg_targets) * torch.log(1-score_neg+eps))
    return loss
    
class HDMI(Loss):
    def __init__(self, eps: float=1e-15):
        super(HDMI, self).__init__()
        self.eps = eps
        self.criterion = nn.BCEWithLogitsLoss()

    def compute(self, score_pos, score_neg, *args, **kwargs) -> torch.FloatTensor:
        loss = hdmi_loss(score_pos, score_neg, self.criterion, self.eps)
        return loss


# In[ ]:




