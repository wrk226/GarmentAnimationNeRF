from lib.config import cfg, args
import numpy as np

import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

def run_dataset():
    from lib.datasets import make_data_loader
    import tqdm

    cfg.train.num_workers = 2
    data_loader = make_data_loader(cfg, is_train=False)
    train_loader = make_data_loader(cfg, is_train=True)

    for batch in tqdm.tqdm(data_loader):
        pass

def visualize_pointcloud(pts):
    import open3d as o3d
    def create_coordinate_frame(size=1, origin=[0, 0, 0]):
        """创建一个Open3D坐标系对象"""
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=origin)
        return coord_frame

    point_cloud = pts
    # 将NumPy数组转换为Open3D的PointCloud对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud.cpu().numpy())

    # 创建坐标轴对象
    coord_frame = create_coordinate_frame()

    # 将坐标轴对象添加到可视化窗口
    o3d.visualization.draw_geometries([pcd, coord_frame])

import torch
def to_cuda(batch):
    if isinstance(batch, tuple) or isinstance(batch, list):
        batch = [to_cuda(b) for b in batch]
        return batch

    for k in batch:
        if isinstance(batch[k], tuple) or isinstance(batch[k], list):
            batch[k] = [b.to('cuda') if torch.is_tensor(b) else b for b in batch[k]]
        elif torch.is_tensor(batch[k]):
            batch[k] = batch[k].to('cuda')
        else:
            continue
    return batch


def run_seen_pose(prefix = 'seen_pose'):
    from lib.datasets import make_data_loader
    from lib.evaluators import make_evaluator
    import tqdm
    import torch
    from lib.networks import make_network
    from lib.utils import net_utils
    from lib.networks.renderer import make_renderer
    torch.cuda.empty_cache()


    network = make_network(cfg).cuda()
    real_epoch = net_utils.load_network(network,
                           cfg.trained_model_dir,
                           resume=cfg.resume,
                           epoch=args.test_epoch)
    network.eval()
    data_loader = make_data_loader(cfg, is_train=False)
    renderer = make_renderer(cfg, network)
    evaluator = make_evaluator(cfg)
    for batch in tqdm.tqdm(data_loader):
        with torch.no_grad():
            batch = to_cuda(batch)
            resolution = 256#512

            batch['prefix'] = f'{prefix}_epoch{real_epoch:04d}_res{resolution}_'
            batch['plot_descriptor'] = False
            if sum(cfg.novel_pose_rotation) !=0.:
                batch['prefix'] += f'{cfg.novel_pose_rotation}'

            print(f'Rendering image of frame {batch["frame_index"].item()} view {batch["cam_ind"].item()}...')
            batch['mode'] = 'test_img'
            output = renderer.render(batch)

            evaluator.evaluate(output, batch)


def run_seen_pose_palette(prefix = 'seen_pose_palette'):
    from lib.datasets import make_data_loader
    from lib.evaluators import make_evaluator
    import tqdm
    import torch
    from lib.networks import make_network
    from lib.utils import net_utils
    from lib.networks.renderer import make_renderer
    torch.cuda.empty_cache()


    network = make_network(cfg).cuda()
    real_epoch = net_utils.load_network(network,
                           cfg.trained_model_dir,
                           resume=cfg.resume,
                           epoch=args.test_epoch)
                                # cfg.test.epoch)
    # network.train()
    network.eval()
    data_loader = make_data_loader(cfg, is_train=False)
    renderer = make_renderer(cfg, network)
    evaluator = make_evaluator(cfg)

    if 'zju' in cfg.train_dataset.data_root:
        colors = [
            np.array([.8, 0.2, 0.2]),  # 红 # 0
            np.array([.8, .8, 0.2]),  # 黄
            np.array([0.2, .8, 0.2]),  # 绿
            np.array([0.2, .8, .8]),  # 青 # 3
            np.array([0.2, 0.2, .8]),  # 蓝
            np.array([.8, 0.2, .8]) # 紫 # 5
        ]
    else:
        colors = [
            np.array([1., 0., 0.]),  # 红 # 0
            np.array([1., 1., 0.]),  # 黄
            np.array([0., 1., 0.]),  # 绿
            np.array([0., 1., 1.]),  # 青 # 3
            np.array([0., 0., 1.]),  # 蓝    v
            np.array([1., 0., 1.]) # 紫 # 5
        ]
    steps_per_color = 10
    repeat_iter = 1 #60
    for batch in tqdm.tqdm(data_loader):
        for ii in range(repeat_iter):
            with torch.no_grad():
                batch = to_cuda(batch)

                resolution = 256#512

                batch['prefix'] = f'{prefix}{args.palette_idx}_epoch{real_epoch:04d}_res{resolution}_'
                if ii>1:
                    batch['prefix'] += f'{ii}'
                batch['plot_descriptor'] = False
                print(f'Rendering image of frame {batch["frame_index"].item()} view {batch["cam_ind"].item()}...')
                batch['mode'] = 'test_img'

                for cidx in [0,3,4]:
                    batch['prefix'] += f'c{cidx}_'
                    curr_color = colors[cidx]
                    print(curr_color)

                    basis_color_val = renderer.net.basis_color.reshape(cfg.num_basis, 3).clamp(0,1).detach().clone()
                    basis_color_val[args.palette_idx] = torch.tensor(torch.from_numpy(curr_color)[None]).cuda()[0]
                    batch['palette_basis_color'] = basis_color_val
                    output = renderer.render(batch)
                    evaluator.evaluate(output, batch)




def run_unseen_pose_palette(prefix = 'unseen_pose_palette'):
    run_seen_pose_palette(prefix=prefix)


def run_first_pose_cloth_rotate(prefix = 'first_pose_cloth_rotate'):
    run_first_pose_rotate(prefix)

def run_unseen_first_pose_rotate(prefix = 'unseen_first_pose_rotate'):
    run_first_pose_rotate(prefix)


def run_first_pose_rotate(prefix = 'first_pose_rotate'):
    from lib.datasets import make_data_loader
    from lib.evaluators import make_evaluator
    import tqdm
    import torch
    from lib.networks import make_network
    from lib.utils import net_utils
    from lib.networks.renderer import make_renderer
    torch.cuda.empty_cache()


    network = make_network(cfg).cuda()
    real_epoch = net_utils.load_network(network,
                           cfg.trained_model_dir,
                           resume=cfg.resume,
                           epoch=args.test_epoch)

    network.eval()
    data_loader = make_data_loader(cfg, is_train=False)
    renderer = make_renderer(cfg, network)
    evaluator = make_evaluator(cfg)
    for batch in tqdm.tqdm(data_loader):
        with torch.no_grad():
            batch = to_cuda(batch)
            resolution = 256#512
            batch['prefix'] = f'{prefix}_epoch{real_epoch:04d}_res{resolution}_rot{batch["extra_rot"].item():04d}_'
            if sum(cfg.novel_pose_rotation) != 0:
                batch['prefix'] += f'{cfg.novel_pose_rotation}'
            batch['plot_descriptor'] = False

            print(f'Rendering image of frame {batch["frame_index"]} view {batch["cam_ind"]}...')
            batch['mode'] = 'test_img'
            output = renderer.render(batch)
            evaluator.evaluate(output, batch)


def run_seen_pose_rotate(prefix = 'seen_pose_rotate'):
    run_seen_pose(prefix=prefix)



def run_unseen_pose(prefix = 'unseen_pose'):
    run_seen_pose(prefix=prefix)

def run_new_pose(prefix = 'new_pose'):
    run_seen_pose(prefix=prefix)

def run_unseen_pose_rotate(prefix = 'unseen_pose_rotate'):
    run_seen_pose(prefix=prefix)



if __name__ == '__main__':
    print(args.type)
    globals()['run_' + args.type]()