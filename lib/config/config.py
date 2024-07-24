from . import yacs
from .yacs import CfgNode as CN
import argparse
import os
import numpy as np
import pprint

cfg = CN()

# experiment name
cfg.exp_name = 'hello'

# network
cfg.point_feature = 9
cfg.distributed = False

# data
cfg.human = 313
cfg.training_view = [0, 6, 12, 18]
cfg.intv = 1
cfg.begin_ith_frame = 0  # the first smpl
cfg.num_train_frame = 1  # number of smpls
cfg.num_render_frame = -1  # number of frames to render
cfg.ith_frame = 0  # the i-th smpl
cfg.frame_interval = 1
cfg.nv = 6890  # number of vertices
cfg.smpl = 'smpl_4views_5e-4'
cfg.vertices = 'vertices'
cfg.params = 'params_4views_5e-4'
cfg.mask_bkgd = True
cfg.sample_smpl = False
cfg.sample_grid = False
cfg.H = 1024
cfg.W = 1024
cfg.add_pointcloud = False

cfg.big_box = False

cfg.rot_ratio = 0.
cfg.rot_range = np.pi / 32

# mesh
# cfg.mesh_th = 50  # threshold of alpha
cfg.mesh_th = 50 #0.05
# task
cfg.task = 'nerf4d'

# gpus
cfg.gpus = list(range(8))
# if load the pretrained network
cfg.resume = True

# epoch
cfg.ep_iter = -1
cfg.save_ep = 100
cfg.save_latest_ep = 1
cfg.eval_ep = 50

# -----------------------------------------------------------------------------
# train
# -----------------------------------------------------------------------------
cfg.train = CN()

cfg.train.dataset = 'CocoTrain'
cfg.train.epoch = 10000
cfg.train.num_workers = 4
cfg.train.collator = ''
cfg.train.batch_sampler = 'default'
cfg.train.sampler_meta = CN({'min_hw': [256, 256], 'max_hw': [480, 640], 'strategy': 'range'})
cfg.train.shuffle = True

# use adam as default
cfg.train.optim = 'adam'
cfg.train.lr = 1e-4
cfg.train.weight_decay = 0

cfg.train.scheduler = CN({'type': 'multi_step', 'milestones': [80, 120, 200, 240], 'gamma': 0.5})

cfg.train.batch_size = 1

cfg.train.acti_func = 'relu'

cfg.train.use_vgg = False
cfg.train.vgg_pretrained = ''
cfg.train.vgg_layer_name = [0,0,0,0,0]

cfg.train.use_ssim = False
cfg.train.use_d = False

# test
cfg.test = CN()
cfg.test.dataset = 'CocoVal'
cfg.test.batch_size = 1
cfg.test.epoch = -1
cfg.test.sampler = 'default'
cfg.test.batch_sampler = 'default'
cfg.test.sampler_meta = CN({'min_hw': [480, 640], 'max_hw': [480, 640], 'strategy': 'origin'})
cfg.test.frame_sampler_interval = 30

# trained model
cfg.trained_model_dir = 'data/trained_model'

# recorder
cfg.record_dir = 'data/record'
cfg.log_interval = 50
cfg.record_interval = 50

# result
cfg.result_dir = 'data/result'

# evaluation
cfg.skip_eval = False
cfg.test_novel_pose = False
cfg.novel_pose_ni = 100
cfg.vis_novel_pose = False
cfg.vis_novel_view = False
cfg.vis_rotate_smpl = False
cfg.vis_mesh = False
cfg.eval_whole_img = False

cfg.fix_random = False
# cfg.fix_random = True

cfg.vis = 'mesh'



# optimize mesh
cfg.optimize_smpl = True

# shape control
cfg.shape_control = [[]]

# clothes change
cfg.upper_body = ''
cfg.lower_body = ''
cfg.upper_body_dir = ''
cfg.lower_body_dir = ''

cfg.use_normal = False
cfg.use_neuse = False
cfg.H = 1024
cfg.W = 1024
cfg.num_latent_code = -1
cfg.erode_edge = True
cfg.box_padding = 0.05

cfg.uv_size = 128#128
# mesh
cfg.mesh = CN()

def parse_cfg(cfg, args):
    if len(cfg.task) == 0:
        raise ValueError('task must be specified')

    # clothes change
    if cfg.upper_body:
        cfg.upper_body_dir = os.path.join(cfg.trained_model_dir, cfg.task, cfg.upper_body)
    if cfg.lower_body:
        cfg.lower_body_dir = os.path.join(cfg.trained_model_dir, cfg.task, cfg.lower_body)

    if cfg.num_latent_code < 0:
        cfg.num_latent_code = cfg.num_train_frame
    # assign the gpus
    # os.environ['CUDA_VISIBLE_DEVICES'] = ', '.join([str(gpu) for gpu in cfg.gpus])
    cfg.trained_model_dir = os.path.join(cfg.trained_model_dir, cfg.task, cfg.exp_name)
    cfg.record_dir = os.path.join(cfg.record_dir, cfg.task, cfg.exp_name)
    cfg.result_dir = os.path.join(cfg.result_dir, cfg.task, cfg.exp_name)
    cfg.local_rank = args.local_rank
    cfg.distributed = cfg.distributed or args.launcher not in ['none']


def make_cfg(args):
    # print("refreshing config")
    with open(args.cfg_file, 'r') as f:
        current_cfg = yacs.load_cfg(f)

    if 'parent_cfg' in current_cfg.keys():
        with open(current_cfg.parent_cfg, 'r') as f:
            parent_cfg = yacs.load_cfg(f)
        cfg.merge_from_other_cfg(parent_cfg)

    cfg.merge_from_other_cfg(current_cfg)
    cfg.merge_from_list(args.opts)

    if cfg.vis_novel_pose:
        cfg.merge_from_other_cfg(cfg.novel_pose_cfg)

    if cfg.vis_novel_view:
        cfg.merge_from_other_cfg(cfg.novel_view_cfg)

    if cfg.vis_rotate_smpl:
        cfg.merge_from_other_cfg(cfg.rotate_smpl_cfg)

    if cfg.vis_mesh:
        cfg.merge_from_other_cfg(cfg.mesh_cfg)

    cfg.merge_from_list(args.opts)

    parse_cfg(cfg, args)
    if 'unseen_pose' in args.type:
        cfg.merge_from_other_cfg(cfg.unseen_pose_cfg)
        # cfg.num_train_frame = min(60, cfg.num_train_frame)
    elif 'seen_pose' in args.type:
        cfg.merge_from_other_cfg(cfg.seen_pose_cfg)
        # cfg.num_train_frame = min(60, cfg.num_train_frame)
    if 'unseen_first_pose' in args.type:
        if 'test_frame' in args:
            cfg.unseen_first_pose_cfg.begin_ith_frame = args.test_frame
        if 'test_cam_pose' in args:
            cfg.test_cam_pose = args.test_cam_pose
        cfg.merge_from_other_cfg(cfg.unseen_first_pose_cfg)
        # cfg.num_train_frame = min(60, cfg.num_train_frame)
    elif 'first_pose' in args.type:
        if 'test_frame' in args:
            cfg.first_pose_cfg.begin_ith_frame = args.test_frame
        if 'test_cam_pose' in args:
            cfg.test_cam_pose = args.test_cam_pose
        cfg.merge_from_other_cfg(cfg.first_pose_cfg)

    if 'new_pose' in args.type:
        cfg.merge_from_other_cfg(cfg.new_pose_cfg)

    if 'first_pose_cloth' in args.type:
        cfg.render_body = False

    if 'rotate' in args.type:
        cfg.rotate_camera = True

    return cfg


parser = argparse.ArgumentParser()
parser.add_argument("--cfg_file", default="configs/default.yaml", type=str)
parser.add_argument('--test', action='store_true', dest='test', default=False)
parser.add_argument("--type", type=str, default="")
parser.add_argument('--det', type=str, default='')
parser.add_argument('--body_suffix', type=str, default='')
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--test_epoch', type=int, default=200)
parser.add_argument('--launcher', type=str, default='none', choices=['none', 'pytorch'])
parser.add_argument('--train_epochs', type=int, default=400)
parser.add_argument('--test_frame', type=int, default=418) #277
parser.add_argument('--test_cam_pose', type=int, default=0) #277
parser.add_argument('--palette_idx', type=int, default=0) #277
parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
args = parser.parse_args()
if len(args.type) > 0:
    cfg.task = "run"
cfg = make_cfg(args)
