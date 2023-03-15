import torch.optim as optim

def get_optimizer(config, model):
    def OPTIMIZER():
        OPTIMIZER = {}
        OPTIMIZER["sgd"] = {}
        OPTIMIZER["sgd"]['Main'] = optim.SGD(
            # filter(lambda p: p.requires_grad, model.parameters()),
            [{'params':[ param for name, param in model.named_parameters() if ('Discriminator' not in name) and (not (config.TRAIN.FREEZE_ENCODER and 'G_Encoder' in name))]}], 
            lr=config.TRAIN.LR,
            momentum=config.TRAIN.MOMENTUM,
            weight_decay=config.TRAIN.WD,
            nesterov=config.TRAIN.NESTEROV
        )
        OPTIMIZER["sgd"]['Dis'] = optim.SGD(
            # filter(lambda p: p.requires_grad, model.parameters()),
            [{'params':[ param for name, param in model.named_parameters() if 'Discriminator' in name]}], 
            lr=config.TRAIN.LR * 0.5,
            momentum=config.TRAIN.MOMENTUM,
            weight_decay=config.TRAIN.WD,
            nesterov=config.TRAIN.NESTEROV
        )
        OPTIMIZER["adam"] = {}
        OPTIMIZER["adam"]['Main'] = optim.Adam(
            [{'params':[ param for name, param in model.named_parameters() if 'Discriminator' not in name]}], 
            lr=config.TRAIN.LR,
            eps=1e-4
        )
        OPTIMIZER["adam"]['Dis'] = optim.Adam(
            [{'params':[ param for name, param in model.named_parameters() if 'Discriminator' in name]}], 
            lr=config.TRAIN.LR * 0.5,
            eps=1e-4
        )
        OPTIMIZER["rmsprop"] = optim.RMSprop(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.TRAIN.LR,
            momentum=config.TRAIN.MOMENTUM,
            weight_decay=config.TRAIN.WD,
        )
        return OPTIMIZER
    return OPTIMIZER()[config.TRAIN.OPTIMIZER]