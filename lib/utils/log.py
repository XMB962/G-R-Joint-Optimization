# 系统工具：创建日志；创建路径; 配置显示; 更新日志信息
# imported by lib.core.function, train

import time
import logging
from pathlib import Path

def get_logger(filename, verbosity=1, name=None):
    '''创建日志'''
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(filename)s][line:%(lineno)d] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

def create_log_folder(cfg, phase='train'):
    '''创建保存模型路径'''
    root_output_dir = Path(cfg.OUTPUT_DIR)
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    dataset = cfg.DATASET.DATASET
    model = cfg.MODEL.NAME
    time_str = time.strftime('%Y-%m-%d-%H-%M')

    if phase == 'train':
        checkpoints_output_dir = root_output_dir / dataset / model / time_str / 'checkpoints'
        print('=> creating {}'.format(checkpoints_output_dir))
        checkpoints_output_dir.mkdir(parents=True, exist_ok=True)

    tensorboard_log_dir = root_output_dir / dataset / model / time_str / 'log'
    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    generate_result = root_output_dir / dataset / model / time_str / 'generate_result'
    print('=> creating {}'.format(generate_result))
    generate_result.mkdir(parents=True, exist_ok=True)

    result = {
            'tb_dir': str(tensorboard_log_dir), 
            'gen_dir': str(generate_result)
        }

    if phase == 'train':
        result['chs_dir'] = str(checkpoints_output_dir)
        
    return result

def display_config(config, logger):
    msg = []
    msg.append('┌{:─^20}─{:─^35}─{:─^25}┐'.format('─','─','─'))
    msg.append('│{: ^82}│'.format('config'))
    msg.append('├{:─^20}┬{:─^35}┬{:─^25}┤'.format('─','─','─'))
    msg.append('│{: ^20}│{: ^35}│{: ^25}│'.format(' ', 'NAME', config.MODEL.NAME))
    msg.append('│{: ^20}│{: ^35}│{: ^25}│'.format('Model', 'NUM_HIDDEN', config.MODEL.NUM_HIDDEN))
    msg.append('│{: ^20}│{: ^35}│{: ^25}│'.format('Parameter', 'RPE', str(config.MODEL.RPE)))
    msg.append('│{: ^20}│{: ^35}│{: ^25}│'.format(' ', 'MULTI_HEAD_LAYER', config.MODEL.NUM_LAYER))
    msg.append('│{: ^20}│{: ^35}│{: ^25}│'.format(' ', 'TREE POS', str(config.MODEL.TREE_POS)))
    msg.append('├{:─^20}┼{:─^35}┼{:─^25}┤'.format('─','─','─'))
    msg.append('│{: ^20}│{: ^35}│{: ^25}│'.format('Dataset', 'IMAGE_SIZE H', config.MODEL.IMAGE_SIZE.H))
    msg.append('│{: ^20}│{: ^35}│{: ^25}│'.format('Setting', 'IMAGE_SIZE W', config.MODEL.IMAGE_SIZE.W))
    msg.append('├{:─^20}┼{:─^35}┼{:─^25}┤'.format('─','─','─'))
    msg.append('│{: ^20}│{: ^35}│{: ^25}│'.format(' ','drop', config.TRAIN.DROPOUT))
    msg.append('│{: ^20}│{: ^35}│{: ^25}│'.format(' ','Learning Rate', config.TRAIN.LR))
    msg.append('│{: ^20}│{: ^35}│{: ^25}│'.format('Train','Weight decay', config.TRAIN.WD))
    msg.append('│{: ^20}│{: ^35}│{: ^25}│'.format(' ','OPTIMIZER', config.TRAIN.OPTIMIZER))
    msg.append('│{: ^20}│{: ^35}│{: ^25}│'.format(' ','MOMENTUM', config.TRAIN.MOMENTUM))
    msg.append('├{:─^20}┼{:─^35}┼{:─^25}┤'.format('─','─','─'))
    for idx,(k,v) in enumerate(config.TRAIN.LOSS.items()):
        if idx==len(config.TRAIN.LOSS.items())//2:
            msg.append('│{: ^20}│{: ^35}│{: ^25}│'.format('alpha', k, v))
        else:
            msg.append('│{: ^20}│{: ^35}│{: ^25}│'.format(' ', k, v))
    msg.append('├{:─^20}┴{:─^35}┴{:─^25}┤'.format('─','─','─'))
    msg.append('│{: ^82}│'.format(config.MSG))
    msg.append('└{:─^20}─{:─^35}─{:─^25}┘'.format('─','─','─'))
    # config.TRAIN.LOSS.R, config.TRAIN.LOSS.G, config.TRAIN.LOSS.E, config.TRAIN.LOSS.J
    for m in msg:
        logger.info(m)


def updata_loss(config, loss, losses, end, batch_size, batch_time, speed):
    '''更新日志信息'''
    if config.MODEL.CRITERION in ['MSE', 'CE']:
        losses.update(float(loss.item()), float(batch_size))
        batch_time.update(float(time.time()-end))
        speed.update(float(batch_size/batch_time.val))
        msg = 'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)    ' \
            'Speed {speed.val:.1f}({speed.avg:.1f}) samples/s    ' \
            'Loss {loss.val:.5f} ({loss.avg:.5f})'.format(loss=losses, batch_time=batch_time, speed=speed)
    elif config.MODEL.CRITERION in ['JOINT']:
        losses.update_dict(loss, float(batch_size))
        batch_time.update(float(time.time()-end))
        speed.update(float(batch_size/batch_time.val))
        msg = 'Time {batch_time.val:.3f}s({batch_time.avg:.3f}s)    ' \
            'Speed {speed.val:.1f}({speed.avg:.1f}) samples/s    '.format(batch_time=batch_time, speed=speed)    
        msg += 'Loss ['
        for k,(s,v) in losses.display.items():
            msg += '{}: {:.4f} ({:.4f})    '.format(k,s,v)
            if len(msg)//200 > msg.count('\n'):
                msg += '\n'
        msg = msg.strip(' ') + ']'
    else:
        msg = 'ERROR KEY | CRITERION = {} MODEL = {}'.format(config.MODEL.CRITERION, config.MODEL.NAME)
        raise NameError(msg)
    return msg, losses
