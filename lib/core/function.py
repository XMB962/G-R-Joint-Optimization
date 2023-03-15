from __future__ import absolute_import
import os
import time
import math
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# utils
from lib.utils.calculate import AverageMeter, levenshtein_distance, is_Inf_or_Nan
from lib.utils.training import compute_loss, pred_forward, updata_model
from lib.utils.log import updata_loss, display_config
from lib.utils.testing import decode_pred, pred_valid

def train(config, train_loader, val_loader, converter, model, criterion, optimizer, device, epoch, logger=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    speed = AverageMeter()
    model.train()
    end = time.time()

    if config.LOCAL_RANK == 0 and logger is not None and epoch%20==0:
        display_config(config, logger)
    for i, package in enumerate(train_loader):
        if (i+1) % 800 == 0 and config.LOCAL_RANK == 0:
            torch.save(model.state_dict(),  os.path.join(config.MODEL.SAVEPATH, "checkpoint_r.pth"))
        if (i+1) % max(2500, len(train_loader)//3) == 0 and config.DATASET.IF_VALID:
            validate(config, val_loader, converter, model, device, logger, istrain=True)
            model.train()
        batch_size = package['image'].size(0)
        # forward
        preds = pred_forward(config, model, package)
        # compute loss
        loss = compute_loss(config, package, preds, criterion)
        # is inf or nan
        if is_Inf_or_Nan(loss):
            continue
        # updata
        updata_model(config, optimizer, loss, model)
        msg, losses = updata_loss(config, loss, losses, end, batch_size, batch_time, speed)
        if (i % min(config.PRINT_FREQ, len(train_loader)//6) == 0 or (i+1) % len(train_loader) == 0) and config.LOCAL_RANK == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t'.format(epoch, i, len(train_loader)) + msg
            logger.info(msg)
        end = time.time()

def validate(config, val_loader, converter, model, device, logger, istrain=False):
    if val_loader is None:
        return 0
    losses = AverageMeter()
    model.eval()
    # WER
    REG = {'N': 0, 'S': 0, 'D': 0, 'I': 0, 'C': 0, 'W': 0}
    n_correct = 0
    num_test_sample = 0
    # results storage
    false_list = []

    with torch.no_grad():
        for i, package in enumerate(val_loader):
            if i == config.TEST.NUM_TEST_BATCH:
                break
            # to device
            inp = package['image'].to(device)
            labels = package['labels']
            length = package['valid_len']
            batch_size = inp.size(0)
            preds = pred_valid(config, model, package)
            # target decode
            text_split = []
            for idx, l in enumerate(length):
                temp_label = list(map(int, labels[idx][0:l].tolist()))
                text_split.append(temp_label)
            # unpack
            prediction = preds['predict']
            fake_image = preds['fake_image']
            # run
            if config.MODEL.NAME in ['recognizer', 'joint']:
                num_test_sample += batch_size
                for r_idx, (pred, target) in enumerate(zip(prediction, text_split)):
                    pred = pred.int().detach().cpu().tolist()
                    pred = decode_pred(config, pred, converter)

                    _, _, map_ = levenshtein_distance(pred, target)
                    for k in map_.keys():
                        if k in REG.keys():
                            REG[k] += map_[k]

                    pred = [converter.num2dict[i] for i in pred]
                    pred = ''.join(pred)

                    target = [converter.num2dict[i] for i in target]
                    target = ''.join(target)

                    n_correct += target == pred
                    if target != pred:
                        false_list.append([pred, target])

                    if r_idx < 5 and (i == 0 or i == len(val_loader)-1) and config.LOCAL_RANK == 0:
                        print('pred  ', pred)
                        print('target', target)
                        print('#'*50)

            if config.MODEL.NAME in ['generator', 'joint']:
                loss = torch.nn.MSELoss()(fake_image, inp)
                losses.update(float(loss.item()))

            if istrain:
                continue

            # visual modules
            from lib.operators.visual import attn_visual_img2str, attn_visual_str2img, attn_map_img2str, attn_map_str2img
            print('Running: {} ~ {}'.format(i*batch_size,(i+1)*batch_size))
            if config.MODEL.NAME in ['generator']:
                attn_visual_str2img(inp, model.dec_attn, preds, labels, i, config, converter)
                attn_map_str2img(inp, model.dec_attn, preds, labels, i, config, converter)
            elif config.MODEL.NAME in ['recognizer']:
                attn_visual_img2str(inp, model.dec_attn, preds, text_split, i, config, converter)
                attn_map_img2str(inp, model.dec_attn, preds, text_split, i, config, converter)
            elif config.MODEL.NAME in ['joint']:
                attn_visual_str2img(inp, model.G_attn, fake_image, labels, i, config, converter)
                # attn_map_str2img(inp, model.G_attn, fake_image, labels, i, config, converter)
                # attn_visual_img2str(inp, model.R_attn, prediction, text_split, i, config, converter)
                # attn_map_img2str(inp, model.R_attn, prediction, text_split, i, config, converter)
                pass
            if i==10:
                break

    if config.MODEL.NAME in ['recognizer', 'joint']:
        accuracy = n_correct / float(num_test_sample)
        is_best = accuracy >= config.TEST.BESTACC
        config.TEST.BESTACC = max(accuracy, config.TEST.BESTACC)
        if config.LOCAL_RANK == 0:
            if is_best and istrain:
                torch.save(model.state_dict(),  os.path.join(config.MODEL.SAVEPATH, "checkpoint_max_{:.2f}.pth".format(accuracy*100)))
            msg = ['Recognizer Test:']
            msg.append("    is best:{}".format(is_best))
            msg.append("    best acc is:{}".format(config.TEST.BESTACC))
            msg.append("    [#correct:{} / #total:{}]".format(n_correct, num_test_sample))
            msg.append('    Test accuray: {:.4f}'.format(accuracy))
            msg.append('    ' + str(REG))
            msg.append('    WER:{:.2f}%'.format(100*(REG['I']+REG['D']+REG['S'])/REG['N']))
            for m in msg:
                logger.info(m)
            if is_best:
                if istrain:
                    pp = open(os.path.join(config.MODEL.SAVEPATH, 'valid_best{:.2f}.txt'.format(accuracy*100)), 'w', encoding='utf-8-sig')
                else:
                    pp = open('test_result.txt', 'w', encoding='utf-8-sig')
                for p, t in false_list:
                    pp.write('pred:   '+p+'\n')
                    pp.write('target: '+t+'\n')
                pp.close()

    if config.MODEL.NAME in ['generator', 'joint']:
        is_best = losses.avg < config.TEST.BESTLOSS
        config.TEST.BESTLOSS = min(losses.avg, config.TEST.BESTLOSS)
        if config.LOCAL_RANK == 0 and logger is not None:
            logger.info('Generator Test:')
            logger.info("    is best:{}".format(is_best))
            logger.info("    best loss is:{}".format(config.TEST.BESTLOSS))
            logger.info('    Test loss: {:.4f}'.format(losses.avg))
            if is_best and istrain:
                torch.save(model.state_dict(),  os.path.join(config.MODEL.SAVEPATH, "checkpoint_min_{:.3f}.pth".format(losses.avg)))
            
        
    return 0

