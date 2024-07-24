import os
import importlib

def make_network(cfg):
    module = cfg.network_module
    path = cfg.network_path
    network = importlib.import_module(module, package=path).Network()
    print('Loading network from: ', path)
    return network
