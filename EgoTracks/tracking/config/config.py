from .stark_defaults import cfg as default_stark_config


CONFIGS = {"STARK": default_stark_config}


def get_cfg(model_type):
    return CONFIGS[model_type].clone()
