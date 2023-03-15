# 创建模型工具： 学习率；加载模型；指定设备
# imported by train

import torch
import torch.distributed as dist
import torch.nn as nn
import os

def get_lr_scheduler(config, optimizer):
    '''自适应学习率回调'''
    last_epoch = config.TRAIN.BEGIN_EPOCH
    for _, sub_optimizer in optimizer.items():
        if isinstance(sub_optimizer, list):
            for opt in sub_optimizer:
                lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    opt, config.TRAIN.LR_STEP,
                    config.TRAIN.LR_FACTOR, last_epoch-1
                )
        else:
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                sub_optimizer, config.TRAIN.LR_STEP,
                config.TRAIN.LR_FACTOR, last_epoch-1
            )
    return lr_scheduler

def get_pre_model(config, model, logger, device):
    '''加载模型'''
    def load_my_state_dict(model, state_dict, local_rank):
        pretrained_dict = state_dict
        model_dict = model.state_dict()
        info = []
        if local_rank == 0:
            info.append('┌{:─^50}┬{:─^15}┬{:─^20}┬{:─^20}┐'.format('─', '─', '─', '─'))
            info.append('│{: ^50}│{: ^15}│{: ^20}│{: ^20}│'.format('layer name', 'load', 'shape', 'message'))
            info.append('│{:─^50}┼{:─^15}┼{:─^20}┼{:─^20}│'.format('─', '─', '─', '─'))
        for k, v in list(model_dict.items()):
            msg = ''
            k_list = [k]
            if 'module.' in k:
                k_list.append(k.replace('module.', ''))
            else:
                k_list.append('module.' + k)
            exists = False
            shape = False
            for kk in k_list:
                if kk in pretrained_dict:
                    if v.shape == pretrained_dict[kk].shape:
                        shape = True
                        model_dict[k] = pretrained_dict[kk]
                    else:
                        msg = 'shape error'
                    exists = True
                    break
            if not exists and msg == '':
                msg = 'not exists'
            if local_rank == 0:
                if exists and shape:
                    flg = '√' 
                    size = '×'.join([str(int(s)) for s in v.shape])
                    msg = ''
                elif msg == 'shape error':
                    flg = msg
                    size = '×'.join([str(int(s)) for s in v.shape])
                    msg = '×'.join([str(int(s)) for s in pretrained_dict[kk].shape])
                elif msg == 'not exists':
                    flg = msg
                    size = ''
                    msg = '×'.join([str(int(s)) for s in v.shape])

                info.append('│{: ^50}│{: ^15}│{: ^20}│{: ^20}│'.format(
                    k.replace('module.', ''), flg, size, msg))
        info.append('└{:─^50}┴{:─^15}┴{:─^20}┴{:─^20}┘'.format('─','─','─','─'))
        if logger is not None and local_rank==0:
            for m in info:
                logger.info(m)
        elif local_rank==0:
            for m in info:
                print(m)
        model.load_state_dict(model_dict, False)
        return model

    local_rank = config.LOCAL_RANK
    if config.TRAIN.FINETUNE.IS_FINETUNE:
        model_state_file = config.TRAIN.FINETUNE.FINETUNE_CHECKPOINIT
        if local_rank == 0 and logger is not None:
            logger.info('load checkpoint from {}'.format(model_state_file))
        if os.path.exists(model_state_file):
            checkpoint = torch.load(model_state_file, map_location=device)
            model = load_my_state_dict(model, checkpoint, local_rank)
        elif local_rank == 0 and logger is not None:
            logger.info(" => no checkpoint found")

        if config.TRAIN.FINETUNE.FREEZE:
            for p in model.cnn.parameters():
                p.requires_grad = False
    return model

def get_device_parallel(config, model):
    '''指定设备GPU/CPU 多卡分布'''
    local_rank = config.LOCAL_RANK
    if not config.GPU:
        device = torch.device("cpu")
        model = model.to(device)
        print('use cpu')
    else:
        if config.GPU_NUM == 1:
            device = torch.device("cuda:{}".format(config.GPUID))
            model = model.to(device)
            print('use gpu')
        else:
            dist.init_process_group(backend='nccl')
            local_rank = torch.distributed.get_rank()
            print('use multi gpu | rank: {}'.format(local_rank))
            torch.cuda.set_device(local_rank)
            device = torch.device('cuda:%d' % local_rank)
            model = model.to(device)
            model = nn.parallel.DistributedDataParallel(
                model, device_ids=[local_rank, ], output_device=0, find_unused_parameters=True)
    return model, device, local_rank
