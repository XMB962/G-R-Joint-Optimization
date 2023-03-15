import yaml
from lib.model_reg import REC_CONFIG_REGISTRY
from easydict import EasyDict as edict

def get_config(name):
    fname = REC_CONFIG_REGISTRY.get(name)
    with open(fname, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return edict(config)
