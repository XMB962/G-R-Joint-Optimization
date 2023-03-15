# 训练工具： 训练前向；计算loss；更新log；更新参数
# imported by lib.core.function

import torch

def compute_loss(config, package, preds, criterion):
    '''计算loss'''
    loss = 0.
    device = torch.device('cuda:%d' % config.LOCAL_RANK)
    # unpack
    inp = package['image'].to(device)
    labels = package['labels'].to(device)
    length = package['valid_len'].to(device)
    Real_or_Fake = package['is_real'].to(device).squeeze()
    Syn_or_Structure = package['is_syn'].to(device).squeeze()
    batch_size = inp.size(0)
    pad_idx = torch.linspace(0, batch_size-1, batch_size).long()
    labels[pad_idx, length.long()] = config.DATASET.END_LABEL
    # run
    if config.MODEL.CRITERION == 'CE':
        loss = criterion(preds, labels.long().view(-1))
    elif config.MODEL.CRITERION == 'MSE':
        loss = criterion(preds, inp)
    elif config.MODEL.CRITERION == 'JOINT':
        loss = criterion(preds, inp, labels, length, Real_or_Fake, Syn_or_Structure)
    else:
        msg = 'ERROR KEY | CRITERION = {} MODEL = {}'.format(config.MODEL.CRITERION, config.MODEL.NAME)
        raise NameError(msg)
    return loss

def pred_forward(config, model, package):
    '''训练前向'''
    device = torch.device('cuda:%d' % config.LOCAL_RANK)
    package['device'] = device
    # unpack
    inp = package['image'].to(device)
    labels = package['labels'].to(device)
    attribute = package['attribute'].to(device)
    tree_pos = package['tree_pos'].to(device)
    length = package['valid_len'].to(device)
    # run
    if config.MODEL.NAME in ['recognizer']:
        preds = model((inp, labels, length, tree_pos))
    elif config.MODEL.NAME in ['generator']:
        preds = model((labels, tree_pos, attribute))
    elif config.MODEL.NAME in ['joint']:
        preds = model(package)
    else:
        msg = 'ERROR MODEL NAME | MODEL = {}'.format(config.MODEL.NAME)
        raise NameError(msg)
    return preds

def updata_model(config, optimizer, loss, model):
    '''更新参数'''
    if config.MODEL.CRITERION in ['CE', 'MSE']:
        optimizer['Main'].zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer['Main'].step()
    elif config.MODEL.CRITERION in ['JOINT']:
        losses = 0
        for k,v in loss.items():
            if k in ['D_R_syn', 'D_P_syn', 'D_P_str']:
                continue
            losses += v * config.TRAIN.LOSS[k]
        optimizer['Main'].zero_grad()
        if config.TRAIN.PATTERN.GAN:
            losses.backward(retain_graph=True)
        else:
            losses.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer['Main'].step()
        if config.TRAIN.PATTERN.GAN:
            losses = config.TRAIN.LOSS['D_R_str'] * loss['D_R_str'] + \
                config.TRAIN.LOSS['D_F_str'] * loss['D_F_str'] * -1 + \
                config.TRAIN.LOSS['D_P_str'] * loss['D_P_str'] + \
                config.TRAIN.LOSS['D_R_syn'] * loss['D_R_syn'] + \
                config.TRAIN.LOSS['D_F_syn'] * loss['D_F_syn'] * -1 + \
                config.TRAIN.LOSS['D_P_syn'] * loss['D_P_syn'] 
            optimizer['Dis'].zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer['Dis'].step()

    else:
        msg = 'ERROR KEY | CRITERION = {} MODEL = {}'.format(config.MODEL.CRITERION, config.MODEL.NAME)
        raise NameError(msg)