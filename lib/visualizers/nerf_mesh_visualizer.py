from lib.utils.if_nerf import voxels
import numpy as np
from lib.config import cfg
import os
import trimesh
from termcolor import colored


class Visualizer:
    def __init__(self):
        result_dir = 'data/animation/{}'.format(cfg.exp_name)
        print(
            colored('the mesh results are saved at {}'.format(result_dir),
                    'yellow'))

    def visualize(self, output, batch, is_smpl=False):
        suffix = ''
        if is_smpl:
            suffix = '_smpl'
        if 'iter_step' in batch.keys():
            prefix = f"iter{batch['iter_step']:06d}_"
        elif 'prefix' in batch.keys():
            prefix = batch['prefix']
        else:
            raise ValueError('Either "prefix" or "iter_step" need to be provided in batch.')
        mesh = trimesh.Trimesh(output['posed_vertex'],
                               output['triangle'],
                               process=False)
        # mesh.show()

        result_dir = cfg.result_dir #os.path.join('data/animation/',cfg.task, cfg.exp_name, 'posed_mesh')
        os.makedirs(result_dir, exist_ok=True)
        frame_index = batch['frame_index'][0].item()
        # result_path = os.path.join(result_dir, f'{prefix}frame{frame_index:04d}{suffix}.npy')
        mesh_path = os.path.join(result_dir, f'{prefix}frame{frame_index:04d}{suffix}.ply')

        # np.save(result_path, output)

        mesh.export(mesh_path)
