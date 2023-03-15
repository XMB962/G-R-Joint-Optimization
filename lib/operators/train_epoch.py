import os
import torch
from lib.core import function
from lib.operators.loss import get_criterion

def clear_history(path):
    ld = os.listdir(path)
    his_files = []
    max_files = []
    min_files = []
    txt_files = []
    for n in ld:
        if 'max' in n:
            max_files.append(n)
        elif '_r' in n:
            pass
        elif 'min' in n:
            min_files.append(n)
        elif '.txt' in n:
            txt_files.append(n)
        else:
            his_files.append(n)

    for files in [his_files, max_files, min_files, txt_files]:
        if len(files)>1:
            dct = {}
            for f in files:
                t = int(os.path.getctime(os.path.join(path, f)))
                dct[t] = os.path.join(path, f)
            for idx,key in enumerate(sorted(dct)):
                if idx == len(files)-1:
                    break
                os.remove(dct[key])
                

def train_and_test(config, train_loader, val_loader, converter, model, optimizer, device, epoch, logger, lr_scheduler):
    if config.LOCAL_RANK == 0:
        logger.info('start epoch {} training'.format(epoch))
    criterion = get_criterion(config)
    function.train(config, train_loader, val_loader, converter, model, criterion, optimizer, device, epoch, logger)

    lr_scheduler.step()
    if epoch % config.TEST_FREQ == 0:
        if config.LOCAL_RANK == 0:
            
            torch.save(
                model.state_dict(),
                os.path.join(config.MODEL.SAVEPATH, "checkpoint_{}.pth".format(epoch))
            )
            logger.info('start epoch {} test'.format(epoch))
            clear_history(config.MODEL.SAVEPATH)
        acc = function.validate(config, val_loader, converter, model, device, logger, istrain=True)
        

def test(config, val_loader, converter, model, device, logger):
    function.validate(config, val_loader, converter, model, device, logger)


def train(config, train_loader, val_loader, converter, model, optimizer, device, epoch, logger, lr_scheduler):
    if config.LOCAL_RANK == 0:
        logger.info('start epoch {} training'.format(epoch))
    criterion = get_criterion(config)
    function.train(config, train_loader, val_loader, converter, model, criterion, optimizer, device, epoch, logger)

    if epoch % config.TEST_FREQ == 0 and config.LOCAL_RANK == 0:
        torch.save(
            model.state_dict(),
            os.path.join(config.MODEL.SAVEPATH, "checkpoint_{}.pth".format(epoch))
        )

    lr_scheduler.step()


def epoch_func(config, train_loader, val_loader, converter, model, optimizer, device, epoch, logger, lr_scheduler):
    def FUNC():
        FUNC = {}
        FUNC['True'] = train_and_test
        FUNC['False'] = train
        return FUNC

    func = FUNC()[str(config.DATASET.IF_VALID)]
    func(config, train_loader, val_loader, converter, model, optimizer, device, epoch, logger, lr_scheduler)

