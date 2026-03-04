#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
from timm.utils import AverageMeter

class SimmimTrainer:
    def __init__(self, config, model, swin_params, optimizer, lr_scheduler, scaler, device):
        super(SimmimTrainer, self).__init__()
        self.device = device
        self.config = config

        self.model = model
        self.optimizer = optimizer
        self.scheduler = lr_scheduler
        self.scaler = scaler
        self.swin_params = swin_params

    def train(self, data_loader, mask, epoch):
        self.model.train()
        self.optimizer.zero_grad()

        loss_meter = AverageMeter()
        num_steps = len(data_loader)
        
        for idx, (batched_raw, _,  _) in enumerate(data_loader):
            raw = batched_raw.cuda(non_blocking=True)

            with torch.amp.autocast(self.device.type, enabled=self.config.ENABLE_AMP):
                loss = self.model(raw, mask)

            is_second_order = hasattr(self.optimizer, 'is_second_order') and self.optimizer.is_second_order
            grad_norm = self.scaler(loss, self.optimizer, clip_grad=self.config.TRAIN.CLIP_GRAD,
                                    parameters=self.swin_params, create_graph=is_second_order,
                                    update_grad=(idx + 1) % self.config.TRAIN.ACCUMULATION_STEPS == 0)
        
            if (idx + 1) % self.config.TRAIN.ACCUMULATION_STEPS == 0:
                self.optimizer.zero_grad()
                self.scheduler.step_update((epoch * num_steps + idx) // self.config.TRAIN.ACCUMULATION_STEPS)
        
            loss_scale_value = self.scaler.state_dict()["scale"]
        

            torch.cuda.synchronize()
            loss_meter.update(loss.item(), raw.size(0))

        return loss_meter        

