# 测试工具： 预测前向；预测结果解码
# imported by lib.core.function
import torch

def pred_valid(config, model, package):
    '''预测前向'''
    pred = None
    device = torch.device('cuda:%d' % config.LOCAL_RANK)
    package['device'] = device
    # unpack
    inp = package['image'].to(device)
    labels = package['labels'].to(device)
    attribute = package['attribute'].to(device)
    tree_pos = package['tree_pos'].to(device)
    # run 
    if config.MODEL.NAME in ['generator']:
        preds = model((labels, tree_pos, attribute))
    elif config.MODEL.NAME in ['recognizer']:
        if config.GPU_NUM == 1:
            preds = model.pred((inp))
        else:
            preds = model.module.pred((inp))
    elif config.MODEL.NAME in ['joint']:
        if config.GPU_NUM == 1:
            preds = model.pred(package)
        else:
            preds = model.module.pred(package)
    else:
        msg = 'ERROR KEY | MODEL = {}'.format(config.MODEL.NAME)
        raise NameError(msg)
    return preds

def decode_pred(config, pred, converter):
    '''预测结果解码'''
    if config.MODEL.NAME in ['recognizer', 'joint']:
        for j in range(len(pred)):
            if pred[j] == config.DATASET.END_LABEL:
                break
        pred = pred[:j]
    else:
        msg = 'ERROR KEY | MODEL = {}'.format(config.MODEL.NAME)
        raise NameError(msg)
    return pred