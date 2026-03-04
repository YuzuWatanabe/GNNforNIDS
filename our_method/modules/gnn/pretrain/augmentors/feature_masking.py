#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from modules.gnn.pretrain.augmentors.augmentor import Graph, Augmentor
from modules.gnn.pretrain.augmentors.functional import drop_feature


class FeatureMasking(Augmentor):
    def __init__(self, pf: float):
        super(FeatureMasking, self).__init__()
        self.pf = pf

    def augment(self, g: Graph) -> Graph:
        x, edge_index, edge_weights = g.unfold()
        x = drop_feature(x, self.pf)
        return Graph(x=x, edge_index=edge_index, edge_weights=edge_weights)

