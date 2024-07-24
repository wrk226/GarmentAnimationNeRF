import os
import importlib


def make_visualizer(cfg):
    module = cfg.visualizer_module
    path = cfg.visualizer_path
    visualizer = importlib.import_module(module, package=path).Visualizer()
    print('Loading visualizer from : {}'.format(path))
    return visualizer
