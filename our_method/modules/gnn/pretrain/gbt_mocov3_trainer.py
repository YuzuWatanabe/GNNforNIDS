# coding: utf-8

# In[ ]:


import torch
import dgl
import os.path as osp
from modules.gnn.pretrain.losses.barlow_twins import BarlowTwins
from modules.swin_transformer.models.mocov3 import adjust_moco_momentum

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

        z, _ = self.encoder(batched_graph, x_batch, self.training)
        z_aug, _ = self.encoder(batch_g_aug, x_aug, self.training)

        return z, z_aug
        
class Trainer:
    def __init__(self, config, gnn_module, mocov3, params, optimizer, scheduler, scaler, device):
        super(Trainer, self).__init__()
        self.device = device
        self.config = config
        
        # グラフデータ拡張
        self.gnn_aug = augmentor.Compose([node_shuffling.NodeShuffling(), feature_masking.FeatureMasking(pf=0.2)])

        # 学習モデル
        self.gnn_encoder = GNNEncoder(encoder=gnn_module, augmentor=self.gnn_aug).to(self.device)
        self.mocov3 = mocov3
        
        self.contrast_model = WithinEmbedContrast(loss=BarlowTwins()).to(self.device) # 損失関数(Barlow Twin)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.params = params

        # loss調整用ハイパーパラメータ
        self.lambda_list = [config.MODEL.LAMBDA1, config.MODEL.LAMBDA2, config.MODEL.LAMBDA3]
        
    def train(self, data_loader, epoch):
        num_steps = len(data_loader)
        loss_meter = AverageMeter()
        m = adjust_moco_momentum(epoch)
        
        for idx, (batched_raw, batched_graph, _) in enumerate(data_loader):
            batched_graph = batched_graph.to(self.device, non_blocking=True)
            batched_raw = batched_raw.to(self.device, non_blocking=True)
            self.gnn_encoder.train()
            self.mocov3.train()
            self.optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast(self.device.type, enabled=self.config.ENABLE_AMP):
                gnn_z, gnn_z_aug = self.gnn_encoder(batched_graph, self.device)
                loss_gnn = self.contrast_model(gnn_z, gnn_z_aug)
                loss_llm = self.mocov3(batched_raw, m)

                # lossの加算に指数移動平均を適用
                total_loss = loss_gnn.mean() + loss_llm

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

