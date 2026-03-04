# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import torch
import torch.distributed as dist

try:
    from torch._six import inf
except:
    from torch import inf

# BarlowTwinsによる事前学習の再開
def load_checkpoint_bt(config, 
                       gnn_model, swin_model, 
                       aug1_linear, aug2_linear,
                       proj_head_gnn, proj_head_llm,
                       optimizer, scheduler, scaler, 
                       logger):
    logger.info(f"==============> Resuming Barlow_Twins from {config.MODEL.RESUME}....................")

    if config.MODEL.RESUME.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            config.MODEL.RESUME, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')

    max_accuracy = 0.0

    # モデルの復元
    if 'gnn_model' in checkpoint:
        msg_gnn = gnn_model.load_state_dict(checkpoint['gnn_model'], strict=False)
        logger.info("GNN Model Load Info: " + str(msg_gnn))
    if 'swin_model' in checkpoint:
        msg_swin = swin_model.load_state_dict(checkpoint['swin_model'], strict=False)
        logger.info("Swin Model Load Info: " + str(msg_swin))

    if 'aug1_linear' in checkpoint:
        aug1_linear = aug1_linear.load_state_dict(checkpoint['aug1_linear'], strict=False)

    if 'aug2_linear' in checkpoint:
        aug2_linear = aug2_linear.load_state_dict(checkpoint['aug2_linear'], strict=False)

    if "proj_head_gnn" in checkpoint:
        proj_head_gnn = proj_head_gnn.load_state_dict(checkpoint['proj_head_gnn'], strict=False)

    if "proj_head_llm" in checkpoint:
        proj_head_gnn = proj_head_gnn.load_state_dict(checkpoint['proj_head_llm'], strict=False)

    if not config.EVAL_MODE:
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            
        if 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])

        if 'scaler' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler'])

        if 'epoch' in checkpoint:
            config.defrost()
            config.TRAIN.START_EPOCH = checkpoint['epoch'] + 1
            config.freeze()
            logger.info(f"=> loaded successfully '{config.MODEL.RESUME}' (epoch {checkpoint['epoch']})")

    del checkpoint
    torch.cuda.empty_cache()

    return max_accuracy

# HDMIによる事前学習の再開
def load_checkpoint_hdmi(config, gnn_module, swin_model, discliminator, optimizer, scheduler, scaler, logger):
    logger.info(f"==============> Resuming Barlow_Twins from {config.MODEL.RESUME}....................")

    if config.MODEL.RESUME.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            config.MODEL.RESUME, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')

    max_accuracy = 0.0

    # モデルの復元
    if 'gnn_model' in checkpoint:
        msg_gnn = gnn_model.load_state_dict(checkpoint['gnn_model'], strict=False)
        logger.info("GNN Model Load Info: " + str(msg_gnn))
    if 'swin_model' in checkpoint:
        msg_swin = swin_model.load_state_dict(checkpoint['swin_model'], strict=False)
        logger.info("Swin Model Load Info: " + str(msg_swin))

    if 'discliminator' in checkpoint:
        msg_discliminator = discliminator.load_state_dict(checkpoint['discliminator'], strict=False)
        
    if not config.EVAL_MODE:
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            
        if 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])

        if 'scaler' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler'])

        if 'epoch' in checkpoint:
            config.defrost()
            config.TRAIN.START_EPOCH = checkpoint['epoch'] + 1
            config.freeze()
            logger.info(f"=> loaded successfully '{config.MODEL.RESUME}' (epoch {checkpoint['epoch']})")

    del checkpoint
    torch.cuda.empty_cache()

    return max_accuracy

# 学習再開用
def load_checkpoint(config, gnn_module, swin_model, optimizer, lr_scheduler, loss_scaler, logger):
    logger.info(f"==============> Resuming form {config.MODEL.RESUME}....................")
    if config.MODEL.RESUME.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            config.MODEL.RESUME, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
    gnn_msg = model.load_state_dict(checkpoint['gnn_model'], strict=False)
    swin_msg = model.load_state_dict(checkpoint['swin_model'], strict=False)
    
    max_accuracy = 0.0
    if not config.EVAL_MODE and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        config.defrost()
        config.TRAIN.START_EPOCH = checkpoint['epoch'] + 1
        config.freeze()
        if 'scaler' in checkpoint:
            loss_scaler.load_state_dict(checkpoint['scaler'])
        logger.info(f"=> loaded successfully '{config.MODEL.RESUME}' (epoch {checkpoint['epoch']})")
        if 'max_accuracy' in checkpoint:
            max_accuracy = checkpoint['max_accuracy']

    del checkpoint
    torch.cuda.empty_cache()
    return max_accuracy

# 事前学習済みモデルの読み込み
def load_pretrained(config, gnn_comb, swin_model, logger):
    logger.info(f"==============> Loading weight {config.MODEL.PRETRAINED} for fine-tuning......")
    checkpoint = torch.load(config.MODEL.PRETRAINED, map_location='cpu')

    gnn_state_dict = checkpoint['gnn_model']
    swin_state_dict = checkpoint['swin_model']

    ### ---- GNN 側の読み込み処理 ---- ###
    gnn_msg = gnn_comb.load_state_dict(gnn_state_dict, strict=False)

    ### ---- Swin 側の読み込み処理 ---- ###
    swin_msg = swin_model.load_state_dict(swin_state_dict, strict=False)
    # 不要なエントリ削除
    keys_to_delete = [k for k in swin_state_dict if any(substr in k for substr in [
        'relative_position_index', 'relative_coords_table', 'attn_mask'])]
    for k in keys_to_delete:
        del swin_state_dict[k]

    swin_state_dict = remove_prefix(swin_state_dict, prefix="encoder.")

    # Bicubic interpolate: relative_position_bias_table
    for k in [k for k in swin_state_dict if "relative_position_bias_table" in k]:
        pretrained = swin_state_dict[k]
        current = swin_model.state_dict()[k]
        if pretrained.shape != current.shape:
            L1, nH1 = pretrained.size()
            L2, nH2 = current.size()
            if nH1 == nH2:
                S1, S2 = int(L1 ** 0.5), int(L2 ** 0.5)
                resized = torch.nn.functional.interpolate(
                    pretrained.permute(1, 0).view(1, nH1, S1, S1),
                    size=(S2, S2), mode='bicubic'
                )
                swin_state_dict[k] = resized.view(nH2, L2).permute(1, 0)
            else:
                logger.warning(f"Skip loading {k} due to head mismatch.")

    # Bicubic interpolate: absolute_pos_embed
    for k in [k for k in swin_state_dict if "absolute_pos_embed" in k]:
        pretrained = swin_state_dict[k]
        current = swin_model.state_dict()[k]
        if pretrained.shape != current.shape:
            _, L1, C1 = pretrained.size()
            _, L2, C2 = current.size()
            if C1 == C2:
                S1, S2 = int(L1 ** 0.5), int(L2 ** 0.5)
                resized = torch.nn.functional.interpolate(
                    pretrained.reshape(-1, S1, S1, C1).permute(0, 3, 1, 2),
                    size=(S2, S2), mode='bicubic'
                )
                swin_state_dict[k] = resized.permute(0, 2, 3, 1).flatten(1, 2)
            else:
                logger.warning(f"Skip loading {k} due to channel mismatch.")

    # 分類ヘッドの形が違えば無視（例: ImageNet22k → ImageNet1k）
    if 'head.bias' in swin_state_dict and 'head.bias' in swin_model.state_dict():
        Nc1 = swin_state_dict['head.bias'].shape[0]
        Nc2 = swin_model.head.bias.shape[0]
        if Nc1 != Nc2:
            logger.warning(f"Reinitializing Swin classifier head: pretrained {Nc1}, current {Nc2}")
            torch.nn.init.constant_(swin_model.head.bias, 0.)
            torch.nn.init.constant_(swin_model.head.weight, 0.)
            del swin_state_dict['head.weight']
            del swin_state_dict['head.bias']

    logger.info(f"=> Pretrained weights loaded successfully from '{config.MODEL.PRETRAINED}'")

    del checkpoint
    torch.cuda.empty_cache()

# BarlowTwinsで学習したモデルの保存
def save_checkpoint_bt(config,
                       gnn_model, swin_model,
                       aug1_linear, aug2_linear,
                       proj_head_gnn, proj_head_llm,
                       optimizer, scheduler,
                       scaler, logger, epoch):
    
    save_state = {'gnn_model': gnn_model.state_dict(),
                  'swin_model': swin_model.state_dict(),
                  'aug1_linear': aug1_linear.state_dict(),
                  'aug2_linear': aug2_linear.state_dict(),
                  'proj_head_gnn': proj_head_gnn.state_dict(),
                  'proj_head_llm': proj_head_llm.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'scheduler': scheduler.state_dict(),
                  'scaler': scaler.state_dict(),
                  'epoch': epoch
                 }

    save_path = os.path.join(config.OUTPUT, f'ckpt_epoch_{epoch}.pth')
    logger.info(f"{save_path} saving......")
    # モデルパラメータの保存
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")

# HDMIで学習したモデルの保存
def save_checkpoint_hdmi(config, gnn_model, swin_model,
                         discliminator, optimizer, scheduler,
                         scaler, logger, epoch):

    save_state = {'gnn_model':gnn_model.state_dict(),
                  'swin_model': swin_model.state_dict(),
                  'discliminator': discliminator.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'scheduler': scheduler.state_dict(),
                  'scaler': scaler.state_dict(),
                  'epoch': epoch
                 }

    save_path = os.path.join(config.OUTPUT, f'ckpt_epoch_{epoch}.pth')
    logger.info(f"{save_path} saving......")
    # モデルパラメータの保存
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")

# simmim - gbtで学習したモデルの保存
def save_checkpoint_simgbt(config, gnn_model, swin_model, 
                           optimizer, lr_scheduler, scaler, 
                           logger, epoch):
    save_state = {'gnn_model':gnn_model.state_dict(),
                  'swin_model': swin_model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'scheduler': lr_scheduler.state_dict(),
                  'scaler': scaler.state_dict(),
                  'epoch': epoch
                 }

    save_path = os.path.join(config.OUTPUT, f'ckpt_epoch_{epoch}.pth')
    logger.info(f"{save_path} saving......")
    # モデルパラメータの保存
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")
    

def save_checkpoint(config, epoch, 
                    swin_model, graph_model, 
                    fusion_linear, max_accuracy, 
                    optimizer, lr_scheduler, loss_scaler, 
                    logger):
    
    save_state = {'swin_model': swin_model.state_dict(),
                  'gnn_model': graph_model.state_dict(),
                  'fusion_linear': fusion_linear.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'scheduler': lr_scheduler.state_dict(),
                  'max_accuracy': max_accuracy,
                  'scaler': loss_scaler.state_dict(),
                  'epoch': epoch,
                  'config': config
                 }

    save_path = os.path.join(config.OUTPUT, f'ckpt_epoch_{epoch}.pth')
    logger.info(f"{save_path} saving......")
    # モデルパラメータの保存
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")

def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def auto_resume_helper(output_dir):
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth')]
    print(f"All checkpoints founded in {output_dir}: {checkpoints}")
    if len(checkpoints) > 0:
        latest_checkpoint = max([os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime)
        print(f"The latest checkpoint founded: {latest_checkpoint}")
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file

# 各GPUで計算したaccとlossを平均化
def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


def ampscaler_get_grad_norm(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        # infになる計算ブロックを計算グラフから削除(infになる計算ブロックを除外)
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
        print("爆発")
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(),
                                                        norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler(init_scale=1024)

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        # 逆伝播開始
        self._scaler.scale(loss).backward(create_graph=create_graph)
        
        # 特定の間隔でgrad_normの計算を実施
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                # （ampの）lossの拡大によって大きくなった勾配を元の大きさに戻す
                self._scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm=1)
            else:
                # （ampの）lossの拡大によって大きくなった勾配を元の大きさに戻す
                self._scaler.unscale_(optimizer)
                norm = ampscaler_get_grad_norm(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)

# 例：PyTorchでの対応
def remove_prefix(state_dict, prefix="encoder."):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            new_state_dict[k[len(prefix):]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict