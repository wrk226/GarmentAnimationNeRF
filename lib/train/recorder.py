from collections import deque, defaultdict
import torch
from tensorboardX import SummaryWriter
import os
from lib.config.config import cfg

from termcolor import colored
import shutil
import numpy as np
import matplotlib.pyplot as plt
from lib.utils.img_utils import get_img
import math
class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20):
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0

    def update(self, value):
        self.deque.append(value)
        self.count += 1
        self.total += value

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque))
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count


class Recorder(object):
    def __init__(self, cfg):
        if cfg.local_rank > 0:
            return

        log_dir = cfg.record_dir
        if not cfg.resume:
            print(colored('remove contents of directory %s' % log_dir, 'red'))
            # os.system('rm -r %s/*' % log_dir)
            if os.path.exists(log_dir):
                shutil.rmtree(log_dir)
            os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)

        # scalars
        self.epoch = 0
        self.step = 0
        self.loss_stats = defaultdict(SmoothedValue)
        self.batch_time = SmoothedValue()
        self.data_time = SmoothedValue()

        # images
        self.image_stats = defaultdict(object)
        # if 'process_' + cfg.task in globals():
        #     self.processor = globals()['process_' + cfg.task]
        # else:
        #     self.processor = None
    def processor(self, image_stats):
        res = int(math.sqrt(cfg.N_rand))
        processed = {}
        if cfg.rendering_2d:
            processed['rgb-GT'] = image_stats['rgb'].squeeze()
            processed['rgb_map-pred'] = image_stats['rgb_map'].squeeze()
        else:
            if image_stats['mask_at_box'].squeeze().shape[0] == 262144:
                select_coord = torch.stack(torch.where(image_stats['mask_at_box'].reshape(512,512)), dim=1)
            else:
                select_coord = image_stats['select_coord'][0]

            if image_stats['mask_at_box'].squeeze().shape[0] == 262144:
                processed['mask_at_box-mask'] = image_stats['mask_at_box'].reshape(512,512,1).to(torch.float32).cpu()
            # elif cfg.patch_sample:
            #     processed['mask_at_box-mask'] = image_stats['mask_at_box'].reshape(cfg.sample_patch_size,cfg.sample_patch_size,1)
            elif cfg.neural_rendering and not cfg.patch_sample:
                if cfg.sample_bound:
                    processed['mask_at_box-mask'] = image_stats['mask_at_box'].reshape(res,res,1).to(torch.float32).cpu()
                else:
                    processed['mask_at_box-mask'] = get_img(image_stats['mask_at_box'][..., None],select_coord, 'b').to(torch.float32).cpu()
            else:
                processed['mask_at_box-mask'] = get_img(image_stats['mask_at_box'],select_coord, 'b').to(torch.float32).cpu()
            processed['acc_map-sum of weight'] = get_img(image_stats['acc_map'],select_coord, 'b')
            # processed['depth_map-depth'] = get_img(image_stats['depth_map'],select_coord, 'b')
            if cfg.neural_rendering and not cfg.patch_sample:
                processed['rgb_map-pred'] = image_stats['rgb_map'].reshape(512,512,3)
                if 'rgb_map_val' in image_stats:
                    processed['rgb_map_palette'] = image_stats['rgb_map_val'].reshape(512,512,3)
                    # processed['basis_map1'] = image_stats['basis_map1'].reshape(512,512,3)
                    # processed['basis_map2'] = image_stats['basis_map2'].reshape(512,512,3)
                    if 'basis_rgb' in image_stats:
                        for basis_idx in range(image_stats['basis_rgb'].shape[2]):
                            processed[f'basis_map{basis_idx}'] = image_stats['basis_rgb'][:,:,basis_idx].reshape(512,512,3)
                if 'rgb_map_hard' in image_stats:
                    processed['rgb_map-hard'] = image_stats['rgb_map_hard'].reshape(512,512,3)
                # processed['rgb-GT'] = image_stats['rgb'].reshape(512,512,3)
                if cfg.sample_bound:
                    processed['rgb-GT'] = image_stats['rgb_gt'].reshape(512,512,3)
                else:
                    processed['rgb-GT'] = get_img(image_stats['rgb'], select_coord, 'b')
            else:
                processed['rgb-GT'] = get_img(image_stats['rgb'], select_coord, 'b')
                processed['rgb_map-pred'] = get_img(image_stats['rgb_map'],select_coord, 'b')
        # processed['disp_map-distance'] = get_img(image_stats['disp_map'],select_coord, 'b')# get some nan

        return processed

    def update_loss_stats(self, loss_dict):
        if cfg.local_rank > 0:
            return
        for k, v in loss_dict.items():
            self.loss_stats[k].update(v.detach().cpu())

    def update_image_stats(self, image_stats):
        if cfg.local_rank > 0:
            return

        image_stats = self.processor(image_stats)
        for k, v in image_stats.items():
            self.image_stats[k] = v.detach().cpu()

    def record(self, prefix, step=-1, loss_stats=None, image_stats=None):
        if cfg.local_rank > 0:
            return

        step = step if step >= 0 else self.step
        loss_stats = loss_stats if loss_stats else self.loss_stats

        for k, v in loss_stats.items():
            if isinstance(v, SmoothedValue):
                self.writer.add_scalar(f'{prefix}/loss-{k}', v.median, step)
            else:
                self.writer.add_scalar(f'{prefix}/loss-{k}', v, step)

        image_stats = self.processor(image_stats) if image_stats else self.image_stats
        for k, v in image_stats.items():
            self.writer.add_image(f'{prefix}/img-{k}', v, step, dataformats='HWC')

    def state_dict(self):
        if cfg.local_rank > 0:
            return
        scalar_dict = {}
        scalar_dict['step'] = self.step
        return scalar_dict

    def load_state_dict(self, scalar_dict):
        if cfg.local_rank > 0:
            return
        self.step = scalar_dict['step']

    def __str__(self):
        if cfg.local_rank > 0:
            return
        loss_state = []
        for k, v in self.loss_stats.items():
            loss_state.append('{}: {:.6f}'.format(k, v.avg))
        loss_state = '  '.join(loss_state)

        recording_state = '  '.join(['epoch: {}', 'step: {}', '{}', 'data: {:.4f}', 'batch: {:.4f}'])
        return recording_state.format(self.epoch, self.step, loss_state, self.data_time.avg, self.batch_time.avg)


def make_recorder(cfg):
    return Recorder(cfg)
