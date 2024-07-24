import os
import importlib


def make_renderer(cfg, network, network_upper=None, network_lower=None):
    module = cfg.renderer_module
    path = cfg.renderer_path
    renderer = importlib.import_module(module, package=path).Renderer(network, network_upper, network_lower)
    print("Loading renderer from {}...".format(path))
    return renderer