#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
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
        h_pos, s = self.encoder(batched_graph, x_batch, self.training)
        h_neg, _ = self.encoder(batch_g_aug, batch_x_aug, self.training)

        return h_pos, h_neg, s

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
        llm_feat_size = config.MODEL.SWIN.EMBED_DIM
        s_feat_size = config.MODEL.GNN.FAST_EMBED_DIM
        
        self.llm_w = nn.Parameter(torch.Tensor(llm_feat_size, 64)).to(device)
        self.llm_proj_w = nn.Parameter(torch.Tensor(llm_feat_size, 64)).to(device)
        self.s_w = nn.Parameter(torch.Tensor(s_feat_size, 64)).to(device)
        self.s_proj_w = nn.Parameter(torch.Tensor(s_feat_size, 64)).to(device)
        self.comb_feat_w1 = nn.Parameter(torch.Tensor(llm_feat_size, llm_feat_size*2)).to(device)
        self.comb_feat_w2 = nn.Parameter(torch.Tensor(llm_feat_size, llm_feat_size)).to(device)
        self.weight_list = [self.llm_w, self.llm_proj_w, self.s_w, self.s_proj_w, self.comb_feat_w1, self.comb_feat_w2]
        self.reset_parameters()

        #self.sig = nn.Sigmoid()
        self.elu = nn.LeakyReLU()

    def reset_parameters(self):
        for weight in self.weight_list:
            nn.init.xavier_uniform_(weight)

    def forward(self, gnn_z_pos, gnn_z_neg, s, llm_h_pos, llm_h_neg):
        weighted_llm_z = torch.matmul(self.llm_w, llm_h_pos)
        weighted_s = torch.matmul(self.s_w, s)
        z_s = self.elu(torch.matmul(self.s_proj_w, s))        
        
        # ===正例===
        h1_score_pos = torch.sum(torch.matmul(gnn_z_pos, weighted_llm_z), dim=1)  # ノード特徴量とLLM特徴量の結合特徴量
        h2_score_pos = torch.sum(torch.matmul(gnn_z_pos, weighted_s), dim=1) # ノード特徴量とグラフ要約ベクトルの結合特徴量
            
        llm_z_pos = self.elu(torch.matmul(self.llm_proj_w, llm_h_pos))
        comb_feat = self.elu(torch.matmul(self.comb_feat_w1, torch.cat([llm_z_pos, z_s],  dim=0)))
        weighted_comb_feat = torch.matmul(self.comb_feat_w2, comb_feat)
        h3_score_pos = torch.sum(torch.matmul(gnn_z_pos, weighted_comb_feat), dim=1) # ノード，グラフサマリ，LLM特徴量の結合特徴量

        # ===負例===
        h1_score_neg = torch.sum(torch.matmul(gnn_z_neg, weighted_llm_z), dim=1)  # ノード特徴量とLLM特徴量の結合特徴量
        h2_score_neg = torch.sum(torch.matmul(gnn_z_neg, weighted_s),dim=1) # ノード特徴量とグラフ要約ベクトルの結合特徴量
            
        llm_z_neg = self.elu(torch.matmul(self.llm_proj_w, llm_h_neg))
        comb_feat = self.elu(torch.matmul(self.comb_feat_w1, torch.cat([llm_z_neg, z_s], dim=0)))
        weighted_comb_feat = torch.matmul(self.comb_feat_w2, comb_feat)
        h3_score_neg = torch.sum(torch.matmul(gnn_z_pos, weighted_comb_feat), dim=1) # ノード，グラフサマリ，LLM特徴量の結合特徴量

        return h1_score_pos, h2_score_pos, h3_score_pos, h1_score_neg, h2_score_neg, h3_score_neg
        
class HDMITrainer:
    def __init__(self, config, gnn_module, llm_module, 
                 discliminator, params, optimizer, 
                 scheduler, scaler, device):

        super(HDMITrainer, self).__init__()
        self.device = device
        self.config = config

        # グラフデータ拡張
        #self.gnn_aug = augmentor.Compose([edge_removing.EdgeRemoving(pe=0.2), feature_masking.FeatureMasking(pf=0.2)])
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

        # loss調整用ハイパーパラメータ
        self.lambda_list = [config.MODEL.LAMBDA1, config.MODEL.LAMBDA2, config.MODEL.LAMBDA3]

        # 移動平均用パラメータ
        self.ema_loss =  [None, None, None]
        self.inv_ema = []
        self.alpha = 0.5
        
        # DWEMA用　前時刻のLoss
        self.loss_t1 = [None, None, None] # 前時刻
        self.loss_t2 = [None, None, None] # 前々時刻

    def ema(self, loss1, loss2, loss3, epoch):
        # update per batch
        ema_rate = []
        losses = [loss1, loss2, loss3]
        for i, loss_i in enumerate(losses):
            val = loss_i.detach()
            if self.ema_loss[i] is None:
                self.ema_loss[i] = val
            else:
                self.ema_loss[i] = self.alpha * val + (1-self.alpha) * self.ema_loss[i]

        if epoch > 5: # 学習初期の重みは乱高下
            # 逆数を用いた動的重みづけ  （平方根で緩和 + clampで安定化）
            for i, ema_loss in enumerate(self.ema_loss):
                if torch.isnan(ema_loss).any():
                    print("self.ema_loss is NaN")
            
            self.inv_ema = torch.stack([1 / torch.sqrt(l + 1e-8) for l in self.ema_loss])
            self.inv_ema = torch.clamp(self.inv_ema, max=10.0)
            self.lambda_list = F.softmax(self.inv_ema, dim=0)
            
            # DWEMA補正レート 
            if None not in self.loss_t1 and None not in self.loss_t2:
                ema_rate = torch.stack([(lt1.detach() + 1e-8) / (lt2.detach() + 1e-8) for lt1, lt2 in zip(self.loss_t1, self.loss_t2)])
            else:
                ema_rate = torch.ones(3)
        else:
            ema_rate = torch.ones(3)

        # loss_t更新
        self.loss_t2 = self.loss_t1.copy()
        self.loss_t1 = [l.detach() for l in losses] # 計算グラフを切って格納
        # 総合loss
        total_loss = sum(ema_rate[i] * self.lambda_list[i] * losses[i] for i in range(3))
        
        return total_loss        
    
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

                h1_pos_score, h2_pos_score, h3_pos_score, h1_neg_score, h2_neg_score, h3_neg_score = self.discliminator(gnn_z_pos, gnn_z_neg, s, llm_h_pos, llm_h_neg)

                loss1 = self.contrast_model(h1_pos_score, h1_neg_score)
                loss2 = self.contrast_model(h2_pos_score, h2_neg_score)
                loss3 = self.contrast_model(h3_pos_score, h3_neg_score)
                    
                # lossの加算に指数移動平均を適用
                #total_loss = self.ema(loss1, loss2, loss3, epoch)
                total_loss = 0.3*loss1 + 0.3*loss2 + 0.3*loss3
            
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

        with torch.no_grad():
            print("epoch", epoch)
            print("score_pos mean/min/max:", h3_pos_score.mean().item(), h3_pos_score.min().item(), h3_pos_score.max().item())
            print("score_neg mean/min/max:", h3_neg_score.mean().item(), h3_neg_score.min().item(), h3_neg_score.max().item())
 
        return loss_meter.avg

