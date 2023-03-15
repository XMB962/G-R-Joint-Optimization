import sys
import argparse
import yaml
import os

import torch
from torch import optim, nn
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter

# operators
from lib.operators.model import get_model
from lib.operators.dataset import get_dataset
from lib.operators.train_epoch import test as epoch_func
from lib.operators.optimizer import get_optimizer
from lib.operators.config import get_config

from lib.utils.model import get_device_parallel, get_pre_model
from lib.utils.log import create_log_folder, get_logger



def parse_arg():
    parser = argparse.ArgumentParser(description="train ocrmodel")

    parser.add_argument('--model', required=True, type=str, default='ocr_model', help='model name', choices=['generator', 'recognizer', 'joint'])
    parser.add_argument('--gpu_num', type=int, default=1, help='gpu_num')
    parser.add_argument('--pre_model', type=str, default='', help='pre-trained model path')
    parser.add_argument('--batchsize', type=int, default=-1, help='batch size')
    # others
    parser.add_argument('--local_rank', type=int, default=0, help='local_rank')
    args = parser.parse_args()

    # with open(args.cfg, 'r') as f:
    config = get_config(args.model)
    ###
    config.MODEL.NAME = args.model
    ###
    config.GPU_NUM = args.gpu_num
    config.LOCAL_RANK = args.local_rank
    ###
    char_file = config.DATASET.DICT_FILE
    with open(char_file, 'r', encoding='utf-8-sig') as f:
        config.MODEL.NUM_CLASS = len(f.readlines())

    if args.pre_model != '':
        config.TRAIN.FINETUNE.FINETUNE_CHECKPOINIT = args.pre_model
        config.TRAIN.FINETUNE.IS_FINETUNE = True
    elif config.TRAIN.FINETUNE.FINETUNE_CHECKPOINIT:
        config.TRAIN.FINETUNE.IS_FINETUNE = True
    else:
        config.TRAIN.FINETUNE.IS_FINETUNE = False
    
    if args.batchsize == -1:
        pass
    else:
        config.TRAIN.BATCH_SIZE_PER_GPU = args.batchsize
    config.MODEL.SAVEPATH = 'valid_txt'
    config.IF_TEST = True
    return config


def main():
    # load config
    config = parse_arg()

    # cudnn
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    # construct face related neural networks
    model = get_model(config)

    output_dict = create_log_folder(config, phase='test')
    logger = get_logger(output_dict['tb_dir']+'/exp.log')
    config.MODEL.RESULT_SAVE_PATH = output_dict['gen_dir']

    # get device and rank
    model, device, local_rank = get_device_parallel(config, model)
    model.eval()
    model = get_pre_model(config, model, None, device)
    
    _, _, val_loader, val_dataset, converter = get_dataset(config, True)

    epoch_func(
        config,
        val_loader,
        converter,
        model,
        device,
        logger
    )

if __name__ == '__main__':

    main()
