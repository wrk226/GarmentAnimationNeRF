import sys

from lib.config import cfg, args
from lib.networks import make_network
from lib.train import make_trainer, make_optimizer, make_lr_scheduler, make_recorder, set_lr_scheduler
from lib.datasets import make_data_loader
from lib.utils.net_utils import load_model, save_model, load_network
from lib.evaluators import make_evaluator
import torch.multiprocessing
import torch
import torch.distributed as dist
import os
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)


if cfg.fix_random:
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(cfg, network):
    trainer = make_trainer(cfg, network)
    optimizer = make_optimizer(cfg, network)
    scheduler = make_lr_scheduler(cfg, optimizer)

    recorder = make_recorder(cfg)
    evaluator = make_evaluator(cfg)

    begin_epoch = load_model(network,
                             optimizer,
                             scheduler,
                             recorder,
                             cfg.trained_model_dir,
                             resume=cfg.resume)
    if begin_epoch == 0 and cfg.pretrained_model != '':
        pretrained_model = torch.load(cfg.pretrained_model, 'cpu')
        filtered_dict = {k: v for k, v in pretrained_model['net'].items() if 'neural_renderer' not in k}
        missing_keys, unexpected_keys = network.load_state_dict(filtered_dict, strict=False)
        print('load pretrained model from {}'.format(cfg.pretrained_model))
        print('missing keys: {}'.format(missing_keys))
        print('unexpected keys: {}'.format(unexpected_keys))

    set_lr_scheduler(cfg, scheduler)
    train_loader = make_data_loader(cfg,
                                    is_train=True,
                                    max_iter=cfg.ep_iter)

    # val_loader = make_data_loader(cfg, is_train=False)
    # torch.autograd.set_detect_anomaly(True)
    for epoch in range(begin_epoch, args.train_epochs):
        if epoch >= cfg.finetune_epoch and 'images_ref' not in train_loader.dataset.img_src:
            train_loader = make_data_loader(cfg,
                                    is_train=True,
                                    max_iter=cfg.ep_iter)
            train_loader.dataset.img_src = 'images_ref_gray' if 'gray' in cfg.img_src else 'images_ref'
        recorder.epoch = epoch
        if cfg.distributed:
            train_loader.batch_sampler.sampler.set_epoch(epoch)

        trainer.train(epoch, train_loader, optimizer, recorder)
        scheduler.step()
        torch.cuda.empty_cache()
        # python empty cache


        if (epoch + 1) % cfg.save_ep == 0 and cfg.local_rank == 0:
            save_model(network, optimizer, scheduler, recorder,
                       cfg.trained_model_dir, epoch+1)

        if (epoch + 1) % cfg.save_latest_ep == 0 and cfg.local_rank == 0:
            save_model(network,
                       optimizer,
                       scheduler,
                       recorder,
                       cfg.trained_model_dir,
                       epoch+1,
                       last=True)
        # if (epoch + 1) % cfg.eval_ep == 0:
        #     trainer.val(epoch, val_loader, evaluator, recorder, vis_mesh=((epoch + 1) % cfg.mesh_ep == 0))
        #     torch.cuda.empty_cache()
        # break

    return network


def test(cfg, network):
    trainer = make_trainer(cfg, network)
    val_loader = make_data_loader(cfg, is_train=False)
    evaluator = make_evaluator(cfg)
    epoch = load_network(network,
                         cfg.trained_model_dir,
                         resume=cfg.resume,
                         epoch=cfg.test.epoch)
    trainer.test(epoch, val_loader, evaluator)


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


def main():
    network = make_network(cfg)
    train(cfg, network)




if __name__ == "__main__":
    main()
