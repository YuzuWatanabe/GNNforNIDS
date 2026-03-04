#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from modules.gnn.pretrain.augmentors.augmentor import Graph, Augmentor

class Identity(Augmentor):
    def __init__(self):
        super(Identity, self).__init__()

    def augment(self, g: Graph) -> Graph:
        return g

