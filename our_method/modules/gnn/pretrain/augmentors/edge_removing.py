#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from modules.gnn.pretrain.augmentors.augmentor import Graph, Augmentor
from modules.gnn.pretrain.augmentors.functional import dropout_adj


class EdgeRemoving(Augmentor):
    def __init__(self, pe: float):
        super(EdgeRemoving, self).__init__()
        self.pe = pe

    def augment(self, g: Graph) -> Graph:
        x, edge_index, edge_weights = g.unfold()
        edge_index, edge_weights = dropout_adj(edge_index, p=self.pe)
        #edge_weights = edge_weights[edge_mask]
        
        return Graph(x=x, edge_index=edge_index, edge_weights=edge_weights)

