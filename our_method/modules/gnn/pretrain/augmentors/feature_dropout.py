#!/usr/bin/env python
# coding: utf-8

# In[2]:


from modules.gnn.pretrain.augmentors.augmentor import Graph, Augmentor
from modules.gnn.pretrain.augmentors.functional import dropout_feature
import dgl

class FeatureDropout(Augmentor):
    def __init__(self, pf: float):
        super(FeatureDropout, self).__init__()
        self.pf = pf

    def augment(self, g: Graph) -> Graph:
        #x, edge_index, edge_weights = g.unfold()
        x = g.ndata['feat']
        src, dst = g.edges()
        edge_index = torch.stack([src, dst], dim=0)

        x = dropout_feature(x, self.pf)
        return Graph(x=x, edge_index=edge_index, edge_weights=edge_weights)


# In[ ]:




