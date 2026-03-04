#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import dgl
import os.path as osp
from modules.gnn.pretrain.losses.barlow_twins import BarlowTwins

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

from tqdm import tqdm
from modules.gnn.pretrain.contrast_model import WithinEmbedContrast
from timm.utils import accuracy, AverageMeter

"""
グラフ描画用（確認用）
"""
import dgl
import torch
import torch.nn.functional as F
import networkx as nx
import matplotlib.pyplot as plt
import os
import random

# グラフ描画
def show_batch_graph(batch_g, num_graphs_to_show=1):
    # バッチを個々のグラフに分解
    graphs = dgl.unbatch(batch_g)
    
    for i in range(min(num_graphs_to_show, len(graphs))):
        g = graphs[i]
        nx_graph = g.to("cpu").to_networkx()
        pos = nx.spring_layout(nx_graph, seed=42)
        
        plt.figure(figsize=(6,6))
        nx.draw(nx_graph, pos, with_labels=True, node_color='skyblue',
                edge_color='gray', node_size=200, font_size=10)
        plt.title(f"Graph {i}")
        plt.axis("off")
        plt.show()

class GNNEncoder(torch.nn.Module):
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

        x_aug = torch.cat(x_aug_list, dim=0).to(device)
        batch_g_aug.ndata["feat"] = x_aug

        z, s = self.encoder(batched_graph, x_batch, self.training)
        z_aug, s_aug = self.encoder(batch_g_aug, x_aug, self.training)

        return z, z_aug, s, s_aug

class LLMEncoder(torch.nn.Module):
    def __init__(self, encoder, mask_generator):
        super(LLMEncoder, self).__init__()
        self.encoder = encoder
        self.mask_generator = mask_generator

    def forward(self, batched_raw, device):
        mask_generator = self.mask_generator
        mask = mask_generator()
        
        z = self.encoder(batched_raw, mask=None)
        z_aug = self.encoder(batched_raw, mask)

        return z, z_aug
        
class GbtTrainer:
    def __init__(self, config, gnn_module, llm_module, aug1_linear, aug2_linear, proj_head_gnn, proj_head_llm, params, optimizer, scheduler, scaler, device):
        super(GbtTrainer, self).__init__()
        self.device = device
        self.config = config
        
        # グラフデータ拡張
        #self.gnn_aug = augmentor.Compose([node_shuffling.NodeShuffling(), feature_masking.FeatureMasking(pf=0.2)])
        self.gnn_aug = augmentor.Compose([edge_removing.EdgeRemoving(pe=0.1), feature_masking.FeatureMasking(pf=0.2)])
        
        # 行列データ拡張(Mask作成)
        self.mask_generator = MaskGenerator(config, mask_ratio=0.2)       

        # 学習モデル
        self.gnn_encoder = GNNEncoder(encoder=gnn_module, augmentor=self.gnn_aug).to(self.device)
        self.llm_encoder = LLMEncoder(encoder=llm_module, mask_generator=self.mask_generator).to(self.device)

        # GNN, LLMの特徴空間調整用
        self.proj_head_gnn = proj_head_gnn
        self.proj_head_llm = proj_head_llm

        # 特徴結合層
        self.aug1_linear = aug1_linear
        self.aug2_linear = aug2_linear
        
        self.contrast_model = WithinEmbedContrast(loss=BarlowTwins()).to(self.device) # 損失関数(Barlow Twin)
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

    """
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

        # ====== loss_t更新 ======
        self.loss_t2 = self.loss_t1.copy()
        self.loss_t1 = [l.detach() for l in losses] # 計算グラフを切って格納

        # ====== 総合loss ======
        total_loss = sum(ema_rate[i] * self.lambda_list[i] * losses[i] for i in range(3))
        
        return total_loss
    """
        
    def train(self, data_loader, epoch):
        num_steps = len(data_loader)
        loss_meter = AverageMeter()
        
        for idx, (batched_raw, batched_graph, _) in enumerate(data_loader):
            batched_graph = batched_graph.to(self.device, non_blocking=True)
            batched_raw = batched_raw.to(self.device, non_blocking=True)
            self.gnn_encoder.train()
            self.llm_encoder.train()
            self.optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast(self.device.type, enabled=self.config.ENABLE_AMP):
                gnn_z, gnn_z_aug, gnn_s, gnn_s_aug = self.gnn_encoder(batched_graph, self.device)
                llm_z, llm_z_aug = self.llm_encoder(batched_raw, self.device)
                
                # project headを通し、L2正規化
                gnn_s = F.normalize(self.proj_head_gnn(gnn_s), dim=1)
                gnn_s_aug = F.normalize(self.proj_head_gnn(gnn_s_aug), dim=1)
                llm_z = F.normalize(self.proj_head_llm(llm_z), dim=1)
                llm_z_aug = F.normalize(self.proj_head_llm(llm_z_aug), dim=1)

                comb_z = self.aug1_linear(torch.cat([gnn_s, llm_z], dim=1))
                comb_z_aug = self.aug2_linear(torch.cat([gnn_s_aug, llm_z_aug], dim=1))
                comb_z = F.normalize(comb_z)
                comb_z_aug = F.normalize(comb_z_aug)
                
                loss1 = self.contrast_model(comb_z, gnn_s_aug)
                loss2 = self.contrast_model(comb_z, llm_z_aug)
                loss3 = self.contrast_model(comb_z, comb_z_aug)

                """
                if idx == num_steps -1:
                    print(f"[Epoch {epoch}] Final batch losses =====")
                    print(f"mixed-graph_loss : {loss1.item():.4f}")
                    print(f"mixed-raw_loss : {loss2.item():.4f}")
                    print(f"mixed-mixed_loss   : {loss3.item():.4f}")
                    # norms
                    print("llm mean norm", llm_z_aug.norm(dim=1).mean().item())
                    print("comb mean norm", comb_z.norm(dim=1).mean().item())
                    
                    # cosine
                    cos = F.cosine_similarity(llm_z_aug, comb_z, dim=1)
                    print("cos mean", cos.mean().item(), "min", cos.min().item(), "max", cos.max().item())
                    
                    # per-dim std
                    print("llm std per-dim", llm_z_aug.std(dim=0)[:10])
                    print("comb std per-dim", comb_z.std(dim=0)[:10])
                """

                # lossの加算に指数移動平均を適用
                #total_loss = self.ema(loss1, loss2, loss3, epoch)
                total_loss = 1.0*loss1+1.0*loss2+1.0*loss3

            # NaN/Inf チェック
            if not torch.isfinite(total_loss):
                print(f"[Error] Non-finite loss detected at step {idx}: {total_loss.item()}")
                raise RuntimeError("Non-finite loss detected")
            
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
