#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from modules.gnn.pretrain.augmentors.augmentor import Graph, Augmentor
from modules.gnn.pretrain.augmentors.functional import permute


class NodeShuffling(Augmentor):
    def __init__(self):
        super(NodeShuffling, self).__init__()

    def augment(self, g: Graph) -> Graph:
        x, edge_index, edge_weights = g.unfold()
        x = permute(x)
        return Graph(x=x, edge_index=edge_index, edge_weights=edge_weights)

