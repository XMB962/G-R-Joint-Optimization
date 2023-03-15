from torch.utils.data import DataLoader
from lib.dataset import get_dataset as load_dataset
from lib.utils.code import strLabelConverter
import torch


def get_train(config, converter):
    train_dataset = load_dataset(config)(config, converter, is_train=True)
    if config.GPU_NUM == 1:
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=config.TRAIN.BATCH_SIZE_PER_GPU,
            shuffle=config.TRAIN.SHUFFLE,
            num_workers=config.WORKERS,
            pin_memory=config.PIN_MEMORY,
        )
    else:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=config.GPU_NUM, rank=config.LOCAL_RANK) #####
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=config.TRAIN.BATCH_SIZE_PER_GPU,
            num_workers=config.WORKERS,
            pin_memory=config.PIN_MEMORY,
            sampler=train_sampler
        )
        
    return train_dataset, train_loader

def get_test(config, converter):
    val_dataset = load_dataset(config)(config, converter, is_train=False)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.TEST.BATCH_SIZE_PER_GPU,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY,
    )
    return val_dataset, val_loader

def get_dataset(config, test_only = False):
    converter = strLabelConverter(config)
    train_dataset, train_loader = None, None
    if not test_only:
        train_dataset, train_loader = get_train(config, converter)
    def valid_set():
        VALID = {}
        val_dataset, val_loader = get_test(config, converter)
        VALID['True'] = [val_dataset, val_loader]
        VALID['False'] = [None, None]
        return VALID
    val_dataset, val_loader = valid_set()[str(config.DATASET.IF_VALID)]
    return train_loader, train_dataset, val_loader, val_dataset, converter

