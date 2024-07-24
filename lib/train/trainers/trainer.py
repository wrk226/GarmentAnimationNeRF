import sys
import time
import datetime
from math import sqrt
import matplotlib.pyplot as plt
import torch
import tqdm
from torch.nn import DataParallel
from lib.config import cfg
import psutil

class Trainer(object):
    def __init__(self, network_wrapper):
        device = torch.device('cuda:{}'.format(cfg.local_rank))
        # device = torch.device('cpu') # for debugging CUDA error
        network_wrapper = network_wrapper.to(device)
        if cfg.distributed:
            network_wrapper = torch.nn.parallel.DistributedDataParallel(
                network_wrapper,
                device_ids=[cfg.local_rank],
                output_device=cfg.local_rank
            )
        self.network_wrapper = network_wrapper
        self.local_rank = cfg.local_rank
        self.device = device
        self.val_data_iterator = None
        self.iter_step = 0
        # print('type(network): ', type(network))
        # print('type(network): ', type(network.renderer.net))

    def reduce_loss_stats(self, loss_stats):
        reduced_losses = {k: torch.mean(v) for k, v in loss_stats.items()}
        return reduced_losses

    def to_cuda(self, batch):
        if isinstance(batch, tuple) or isinstance(batch, list):
            batch = [self.to_cuda(b) for b in batch]
            return batch

        for k in batch:
            if isinstance(batch[k], tuple) or isinstance(batch[k], list):
                batch[k] = [b.to(self.device) if torch.is_tensor(b) else b for b in batch[k]]
            elif torch.is_tensor(batch[k]):
                batch[k] = batch[k].to(self.device)
            else:
                continue
        return batch

    def add_iter_step(self, batch, iter_step):
        if isinstance(batch, tuple) or isinstance(batch, list):
            for batch_ in batch:
                self.add_iter_step(batch_, iter_step)

        if isinstance(batch, dict):
            batch['iter_step'] = iter_step
            self.iter_step = iter_step

    def train(self, epoch, data_loader, optimizer, recorder):
        max_iter = len(data_loader)
        self.network_wrapper.train()
        end = time.time()
        for iteration, batch in enumerate(data_loader):


            batch['mode'] = 'train'
            batch['epoch'] = epoch
            data_time = time.time() - end
            iteration = iteration + 1

            batch = self.to_cuda(batch)
            self.add_iter_step(batch, epoch * max_iter + iteration)
            output, loss, loss_stats, image_stats = self.network_wrapper(batch)
            # training stage: loss; optimizer; scheduler
            optimizer.zero_grad()
            loss = loss.mean()
            loss.backward()
            if cfg.clip_gradient:
                torch.nn.utils.clip_grad_value_(self.network_wrapper.parameters(), 40)
            optimizer.step()
            if cfg.local_rank > 0:
                continue

            # data recording stage: loss_stats, time, image_stats
            recorder.step += 1

            loss_stats = self.reduce_loss_stats(loss_stats)
            recorder.update_loss_stats(loss_stats)

            batch_time = time.time() - end
            end = time.time()
            recorder.batch_time.update(batch_time)
            recorder.data_time.update(data_time)
            if iteration % cfg.log_interval == 0 or iteration == (max_iter - 1):
                # print training state
                eta_seconds = recorder.batch_time.global_avg * (max_iter - iteration)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                lr = optimizer.param_groups[0]['lr']
                memory = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0

                training_state = '  '.join(['eta: {}', '{}', 'lr: {:.6f}', 'max_mem: {:.0f}'])
                training_state = training_state.format(eta_string, str(recorder), lr, memory)
                print(training_state)
            # if torch.cuda.memory_reserved(0)/ (1024**3)>20:
            #     torch.cuda.empty_cache()

            if True or iteration % cfg.record_interval == 0 or iteration == (max_iter - 1):
                # record loss_stats and image_dict
                recorder.update_image_stats(image_stats)
                recorder.record('train')


    def val(self, epoch, data_loader, evaluator=None, recorder=None, vis_mesh=False):
        try:
            batch = next(self.val_data_iterator)
        except:
            self.val_data_iterator = iter(data_loader)
            batch = next(self.val_data_iterator)

        self.network_wrapper.eval()
        torch.cuda.empty_cache()
        val_loss_stats = {}

        batch = self.to_cuda(batch)
        batch['iter_step'] = 999999
        self.add_iter_step(batch, self.iter_step)
        with torch.no_grad():
            batch['mode'] = 'test_img'
            output, loss, loss_stats, image_stats = self.network_wrapper(batch)
            if evaluator is not None:
                evaluator.evaluate(output, batch)

        loss_stats = self.reduce_loss_stats(loss_stats)
        for k, v in loss_stats.items():
            val_loss_stats.setdefault(k, 0)
            val_loss_stats[k] += v
        loss_state = []
        for k in val_loss_stats.keys():
            loss_state.append('{}: {:.4f}'.format(k, val_loss_stats[k]))
        print(loss_state)


        if evaluator is not None:
            result = evaluator.summarize()
            val_loss_stats.update(result)

        if recorder:
            recorder.record('val', epoch, val_loss_stats, image_stats)
        # vis mesh
        if vis_mesh:
            batch['mode'] = 'test_mesh'
            self.network_wrapper.renderer.get_mesh(batch)
            if hasattr(self.network_wrapper.net, 'curr_smpl_wverts'):
                self.network_wrapper.renderer.get_smpl_mesh(batch)

    def test(self, epoch, data_loader, evaluator=None, recorder=None, vis_mesh=False):
        self.network_wrapper.eval()
        torch.cuda.empty_cache()
        val_loss_stats = {}
        for batch in tqdm.tqdm(data_loader):
            batch = self.to_cuda(batch)
            self.add_iter_step(batch, self.iter_step)
            with torch.no_grad():
                output, loss, loss_stats, image_stats = self.network_wrapper(batch)
                if evaluator is not None:
                    evaluator.evaluate(output, batch)

            loss_stats = self.reduce_loss_stats(loss_stats)
            for k, v in loss_stats.items():
                val_loss_stats.setdefault(k, 0)
                val_loss_stats[k] += v
            loss_state = []
            for k in val_loss_stats.keys():
                loss_state.append('{}: {:.4f}'.format(k, val_loss_stats[k]))
            print(loss_state)

            if evaluator is not None:
                result = evaluator.summarize()
                val_loss_stats.update(result)

            if recorder:
                recorder.record('val', epoch, val_loss_stats, image_stats)

            # vis mesh
            # if vis_mesh:
            #     self.network.get_mesh(batch)
            #     self.network.get_smpl_mesh(batch)
            #     vis_mesh = False

    def test_old(self, epoch, data_loader, evaluator=None, recorder=None):
        self.network_wrapper.eval()
        torch.cuda.empty_cache()
        val_loss_stats = {}
        data_size = len(data_loader)
        for batch in tqdm.tqdm(data_loader):
            batch = self.to_cuda(batch)
            with torch.no_grad():
                output, loss, loss_stats, image_stats = self.network_wrapper(batch)
                if evaluator is not None:
                    evaluator.evaluate(output, batch)

            loss_stats = self.reduce_loss_stats(loss_stats)
            for k, v in loss_stats.items():
                val_loss_stats.setdefault(k, 0)
                val_loss_stats[k] += v

        loss_state = []
        for k in val_loss_stats.keys():
            val_loss_stats[k] /= data_size
            loss_state.append('{}: {:.4f}'.format(k, val_loss_stats[k]))
        print(loss_state)

        if evaluator is not None:
            result = evaluator.summarize()
            val_loss_stats.update(result)

        if recorder:
            recorder.record('val', epoch, val_loss_stats, image_stats)
