#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as DeepGraphInfomax
import dgl
import os.path as osp
import math

from modules.gnn.pretrain.losses.hdmi import HDMI
from modules.gnn.pretrain.augmentors import augmentor
from modules.gnn.pretrain.augmentors import edge_adding
from modules.gnn.pretrain.augmentors import edge_attr_masking
from modules.gnn.pretrain.augmentors import edge_removing
from modules.gnn.pretrain.augmentors import feature_dropout
from modules.gnn.pretrain.augmentors import feature_masking
from modules.gnn.pretrain.augmentors import functional
from modules.gnn.pretrain.augmentors import identity
from modules.gnn.pretrain.augmentors import markov_diffusion
from modules.gnn.pretrain.augmentors import node_dropout
from modules.gnn.pretrain.augmentors import node_shuffling
from modules.gnn.pretrain.augmentors import ppr_difusion
from modules.gnn.pretrain.augmentors import rw_sampling
from data.build_pt import MaskGenerator

from modules.gnn.pretrain.contrast_model import DifferentExtendContrast

from timm.utils import accuracy, AverageMeter

class GNNEncoder(nn.Module):
    def __init__(self, encoder, augmentor):
        super(GNNEncoder, self).__init__()
        self.encoder = encoder
        self.augmentor = augmentor
        self.training = True

    def forward(self, batched_graph, device):
        aug = self.augmentor
        g_list = dgl.unbatch(batched_graph)
        g_aug_list = []
        x_aug_list = []

        x_batch = batched_graph.ndata["feat"]

        for g in g_list:
            x = g.ndata["feat"]
            num_nodes = g.num_nodes()
            src, dst = g.edges()
            edge_index = torch.stack([src, dst], dim=0)
            edge_attr = None

            g_aug, x_aug = aug(x, edge_index, edge_attr, num_nodes)
            g_aug_list.append(g_aug)
            x_aug_list.append(x_aug)

        batch_g_aug = dgl.batch(g_aug_list)
        batch_x_aug = torch.cat(x_aug_list, dim=0).to(device)
        batch_g_aug.ndata["feat"] = batch_x_aug

        # 最終層のノード特徴量とグラフ要約ベクトルの作成
        h_pos, s_pos = self.encoder(batched_graph, x_batch, self.training)
        h_neg, _ = self.encoder(batch_g_aug, batch_x_aug, self.training)

        return h_pos, h_neg, s_pos

class LLMEncoder(nn.Module):
    def __init__(self, encoder, augmentor):
        super(LLMEncoder, self).__init__()
        self.encoder = encoder
        self.augmentor = augmentor

    def forward(self, batched_raw, device):
        mask = self.augmentor()
        
        h_pos = self.encoder(batched_raw, mask=None)
        h_neg = self.encoder(batched_raw, mask)

        return h_pos, h_neg

class Discliminator(nn.Module):
    def __init__(self, config, device):
        super(Discliminator, self).__init__()
        s_feat_size = config.MODEL.GNN.FAST_EMBED_DIM
        self.s_w = nn.Parameter(torch.Tensor(s_feat_size, 64)).to(device)
        self.reset_parameters()
        self.elu = nn.LeakyReLU()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.s_w)

    def forward(self, gnn_z_pos, gnn_z_neg, s):
        s = self.elu(torch.matmul(self.s_w, s))       
        
        # ===正例===
        h_score_pos = torch.sum(torch.matmul(gnn_z_pos, s), dim=1) # ノード特徴量とグラフ要約ベクトルの結合特徴量

        # ===負例===
        h_score_neg = torch.sum(torch.matmul(gnn_z_neg, s),dim=1) # ノード特徴量とグラフ要約ベクトルの結合特徴量
            
        return h_score_pos, h_score_neg
        
class Trainer:
    def __init__(self, config, gnn_module, llm_module, 
                 discliminator, params, optimizer, 
                 scheduler, scaler, device):

        super(Trainer, self).__init__()
        self.device = device
        self.config = config

        # グラフデータ拡張
        self.gnn_aug = augmentor.Compose([node_shuffling.NodeShuffling(), feature_masking.FeatureMasking(pf=0.2)])

        # 行列データ用のマスク
        self.mask_generator = MaskGenerator(config, mask_ratio=0.2)

        # エンコーダ
        self.gnn_encoder = GNNEncoder(encoder=gnn_module, augmentor=self.gnn_aug).to(device)
        self.llm_encoder = LLMEncoder(encoder=llm_module, augmentor=self.mask_generator).to(device)

        self.discliminator = discliminator

        # 損失関数の設定
        self.contrast_model = DifferentExtendContrast(loss=HDMI()).to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.params = params
    
    def train(self, data_loader, epoch):
        num_steps = len(data_loader)
        loss_meter = AverageMeter()

        for idx, (batched_raw, batched_graph, _) in enumerate(data_loader):
            batched_graph = batched_graph.to(self.device, non_blocking=True)
            batched_raw = batched_raw.to(self.device, non_blocking =True)
            self.gnn_encoder.train()
            self.llm_encoder.train()
            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(self.device.type, enabled=self.config.ENABLE_AMP):
                gnn_z_pos, gnn_z_neg, s = self.gnn_encoder(batched_graph, self.device)
                llm_h_pos, llm_h_neg = self.llm_encoder(batched_raw, self.device)

                h_pos_score,  h_neg_score = self.discliminator(gnn_z_pos, gnn_z_neg, s)

                loss_gnn = self.contrast_model(h_pos_score, h_neg_score)
                loss_llm = F.l1_loss(llm_h_pos, llm_h_neg)
                    
                # lossの加算に指数移動平均を適用
                total_loss = 0.9*loss_llm + 0.9*loss_gnn
            
            is_second_order = hasattr(self.optimizer, 'is_second_order') and self.optimizer.is_second_order

            grad_norm = self.scaler(total_loss, self.optimizer, clip_grad=self.config.TRAIN.CLIP_GRAD,
                                parameters=self.params, create_graph=is_second_order,
                                update_grad=(idx + 1) % self.config.TRAIN.ACCUMULATION_STEPS == 0)

            if (idx + 1) % self.config.TRAIN.ACCUMULATION_STEPS == 0:
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step_update((epoch * num_steps + idx) // self.config.TRAIN.ACCUMULATION_STEPS)
        
            loss_scale_value = self.scaler.state_dict()["scale"]

            torch.cuda.synchronize()
            loss_meter.update(total_loss.item(), batched_graph.batch_size)
 
        return loss_meter.avg

