from .rnn import RnnModule

MODULES = {"rnn": RnnModule}


def get_model_names():
    return MODULES.keys()


def get_model_type(name):
    return MODULES[name]
