# GarmentAnimationNeRF

## Introduction

This repository contains the implemetation of [Garment Animation NeRF with Color Editing](https://github.com/wrk226/GarmentAnimationNeRF) proposed in SCA 2024.

## Prerequisites
- Download code & pre-trained model:
Git clone the code by:
```
git clone https://github.com/wrk226/GarmentAnimationNeRF $ROOTPATH
```
The pretrained model and dataset can be found from [here](https://drive.google.com/drive/folders/1IVb8PZ3dC2KsnF9ORE7fnzg4xpryRDQn?usp=sharing).
- Install packages:
```
conda create --name ganerf python=3.9
conda activate ganerf
pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116
pip install git+https://github.com/facebookresearch/pytorch3d.git@v0.7.2
pip install matplotlib
pip install psutil
pip install opencv-python
pip install open3d
pip install scikit-image==0.19.3
pip install kornia
pip install spconv-cu116
pip install chumpy
```

## Train
Run the following command from the `$ROOTPATH` directory:
```
python train_net.py --cfg_file configs/zju_mocap_exp/samba-multilayer_16v-2d-3d-v2-2.yaml --train_epochs 400 exp_name samba-multilayer_16v-2d-3d-v2-2 resume True
```
## Test
Run the following command from the `$ROOTPATH` directory:
```
python run.py --type seen_pose --cfg_file configs/zju_mocap_exp/samba-multilayer_16v-2d-3d-v2-2.yaml --test_epoch 400 exp_name samba-multilayer_16v-2d-3d-v2-2
python run.py --type unseen_pose --cfg_file configs/zju_mocap_exp/samba-multilayer_16v-2d-3d-v2-2.yaml --test_epoch 400 exp_name samba-multilayer_16v-2d-3d-v2-2
```

