from lib.utils.if_nerf import voxels
import numpy as np
from lib.config import cfg
import os
from termcolor import colored


class Visualizer:
    def __init__(self):
        result_dir = os.path.join(cfg.result_dir, 'mesh')
        print(colored('the mesh results are saved at {}'.format(result_dir), 'yellow'))


    def visualize(self, output, batch):
        mesh = output['mesh']
        # mesh.show()

        result_dir = os.path.join(cfg.result_dir, 'mesh')
        # os.system('mkdir -p {}'.format(result_dir))
        os.makedirs(result_dir, exist_ok=True)
        i = batch['frame_index'].item()
        result_path = os.path.join(result_dir, '{:04d}.ply'.format(i))
        mesh.export(result_path)
