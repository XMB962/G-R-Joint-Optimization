from ._GEN import _GEN
def get_dataset(config):
    if config.DATASET.DATASET == "general":
        return _GEN
    else:
        raise NotImplemented()