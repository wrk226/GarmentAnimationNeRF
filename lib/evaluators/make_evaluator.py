import os
import importlib

def _evaluator_factory(cfg):
    module = cfg.evaluator_module
    path = cfg.evaluator_path
    print("Loading evaluator from {}...".format(path))
    evaluator = importlib.import_module(module, package=path).Evaluator()
    return evaluator


def make_evaluator(cfg):
    if cfg.skip_eval:
        return None
    else:
        return _evaluator_factory(cfg)
