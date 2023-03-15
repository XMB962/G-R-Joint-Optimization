import sys
import argparse
import torch.backends.cudnn as cudnn

# operators
from lib.operators.model import get_model
from lib.operators.dataset import get_dataset
from lib.operators.train_epoch import epoch_func
from lib.operators.optimizer import get_optimizer
from lib.operators.config import get_config
from lib.utils.model import get_lr_scheduler, get_device_parallel, get_pre_model
from lib.utils.log import create_log_folder, get_logger

def parse_arg():
    parser = argparse.ArgumentParser(description="train model")

    parser.add_argument('--model', required=True, type=str, default='model', help='model name', choices=['generator', 'recognizer', 'joint'])
    parser.add_argument('--gpu_num', type=int, default=1, help='gpu_num')
    parser.add_argument('--pre_model', type=str, default='', help='pre-trained model path')
    parser.add_argument('--batchsize', type=int, default=-1, help='batch size')
    # others
    parser.add_argument('--local_rank', type=int, default=0, help='local_rank')
    parser.add_argument('--msg', type=str, default=' ', help='extra message')
    args = parser.parse_args()

    # init
    config = get_config(args.model)
    # num class
    char_file = config.DATASET.DICT_FILE
    with open(char_file, 'r', encoding='utf-8-sig') as f:
        config.MODEL.NUM_CLASS = len(f.readlines())
    # config
    config.MODEL.NAME = args.model
    config.GPU_NUM = args.gpu_num
    config.LOCAL_RANK = args.local_rank
    config.MSG = args.msg

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

    # get device and rank
    model, device, local_rank = get_device_parallel(config, model)

    # create output folder
    logger = None
    if local_rank == 0:
        output_dict = create_log_folder(config, phase='train')
        logger = get_logger(output_dict['tb_dir']+'/exp.log')
        config.MODEL.SAVEPATH = output_dict['chs_dir']

    optimizer = get_optimizer(config, model)
    lr_scheduler = get_lr_scheduler(config, optimizer)
    model = get_pre_model(config, model, logger, device)

    train_loader, train_dataset, val_loader, val_dataset, converter = get_dataset(config)

    # start training
    if local_rank == 0:
        logger.info('start training!')

    for epoch in range(config.TRAIN.END_EPOCH):
        epoch_func(
            config,
            train_loader,
            val_loader,
            converter,
            model,
            optimizer,
            device,
            epoch,
            logger,
            lr_scheduler,
        )
        # break

if __name__ == '__main__':
    main()
