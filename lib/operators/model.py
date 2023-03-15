from lib.model_reg import REC_MODEL_REGISTRY

def get_model(config):
    name = config.MODEL.NAME
    model = REC_MODEL_REGISTRY.get(name).get_model(config)
    return model
