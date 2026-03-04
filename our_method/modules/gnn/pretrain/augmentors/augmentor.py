#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#from __future__ import annotations

import torch
from abc import ABC, abstractmethod
from typing import Optional, Tuple, NamedTuple, List
import dgl

# グラフ構造の抽象化（グラフ構造自体ではなく、その構成要素をクラス）
class Graph(NamedTuple):
    x: torch.FloatTensor
    edge_index: torch.LongTensor
    edge_weights: Optional[torch.FloatTensor]

    # 保持しているグラフ情報を返す
    def unfold(self) -> Tuple[torch.FloatTensor, torch.LongTensor, Optional[torch.FloatTensor]]:
        return self.x, self.edge_index, self.edge_weights


class Augmentor(ABC):
    """Base class for graph augmentors."""
    def __init__(self):
        pass

    @abstractmethod
    def augment(self, g: Graph) -> Graph:
        raise NotImplementedError(f"GraphAug.augment should be implemented.")

    # aug(x, edge_index, edge_weight) のように呼び出すと、Graph を作って augment() を適用
    def __call__(
        self, x: torch.FloatTensor,
        edge_index: torch.LongTensor, 
        edge_weight: Optional[torch.FloatTensor],
        num_nodes: int
        ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:

        # Composeのaugmentを呼びだし
        return self.augment(Graph(x, edge_index, edge_weight), num_nodes)

class Compose(Augmentor):
    def __init__(self, augmentors: List[Augmentor]):
        super(Compose, self).__init__()
        self.augmentors = augmentors

    # augmentリスト (edge-removing, feature-removing)を順番に処理
    def augment(self, g, num_nodes):
        # augmentリストの各処理(augment関数)を実施
        for aug in self.augmentors:
            g = aug.augment(g)
        
        # 削除後のedge_indexからDGL形式の無向グラフを再構築
        g_aug, x_aug = self.const_dglg(g, num_nodes)
        return g_aug, x_aug

    # グラフ再構築
    def const_dglg(self, g, num_nodes):
        # unfoldでノード特徴やedge_indexを取得（PyG風のAPIなら）
        x, edge_index, _ = g.unfold()

        # 使用されているノードのみに絞る
        unique_nodes, new_indices = torch.unique(edge_index, return_inverse=True)
        x_aug = x[unique_nodes]

        # edge_index を再マップ
        edge_index_remapped = new_indices.view(edge_index.shape)

        # 対称化（必要なら）
        symmetric_index = self.make_symmetric_edge_index(edge_index_remapped)

        # 個別グラフの構築
        #g_aug = dgl.graph((edge_index_remapped[0], edge_index_remapped[1]), num_nodes=len(unique_nodes))
        g_aug = dgl.graph((symmetric_index[0], symmetric_index[1]), num_nodes=len(unique_nodes))
        g_aug = g_aug.to('cpu')
        g_aug = dgl.add_self_loop(g_aug)  # 自己ループを全ノードに追加
        g_aug = dgl.to_bidirected(g_aug)   # 双方向エッジに変換

        return g_aug.to('cuda'), x_aug

    # エッジ(i,j)削除後、無向グラフの対称性を維持するために、(j,i)成分を削除
    def make_symmetric_edge_index(self, edge_index: torch.Tensor):
        rev_edge_index = edge_index[[1, 0], :]  # reverse (j, i)
        combined = torch.cat([edge_index, rev_edge_index], dim=1)
    
        # 重複エッジを削除
        combined = combined.T
        combined = torch.unique(combined, dim=0)
        
        return combined.T


class RandomChoice(Augmentor):
    def __init__(self, augmentors: List[Augmentor], num_choices: int):
        super(RandomChoice, self).__init__()
        assert num_choices <= len(augmentors)
        self.augmentors = augmentors
        self.num_choices = num_choices

    def augment(self, g: Graph) -> Graph:
        num_augmentors = len(self.augmentors)
        perm = torch.randperm(num_augmentors)
        idx = perm[:self.num_choices]
        for i in idx:
            aug = self.augmentors[i]
            g = aug.augment(g)
        return g

