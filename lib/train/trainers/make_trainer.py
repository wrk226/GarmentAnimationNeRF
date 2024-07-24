from .trainer import Trainer
import importlib

def _wrapper_factory(cfg, network):
    module = cfg.trainer_module
    path = cfg.trainer_path
    network_wrapper = importlib.import_module(module, package=path).NetworkWrapper(network)
    print('Loading network wrapper from: ', path)
    return network_wrapper


def make_trainer(cfg, network):
    network_wrapper = _wrapper_factory(cfg, network)
    return Trainer(network_wrapper)
