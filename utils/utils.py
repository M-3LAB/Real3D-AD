import numpy as np
import random
import torch
from torchvision import transforms
from PIL import ImageFilter
import argparse

def set_seeds(seed: int = 0) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

class KNNGaussianBlur(torch.nn.Module):
    def __init__(self, radius : int = 4):
        super().__init__()
        self.radius = radius
        self.unload = transforms.ToPILImage()
        self.load = transforms.ToTensor()
        self.blur_kernel = ImageFilter.GaussianBlur(radius=4)

    def __call__(self, img):
        map_max = img.max()
        final_map = self.load(self.unload(img[0] / map_max).filter(self.blur_kernel)) * map_max
        return final_map

def get_args_point_mae():
    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('--method_name', default='Point_MAE', type=str, 
                        choices=['DINO', 'Point_MAE', 'Fusion', 'DINO+Point_MAE', 'DINO+Point_MAE+Fusion', 'DINO+Point_MAE+add'],
                        help='Anomaly detection modal name.')
    parser.add_argument('--max_sample', default=400, type=int,
                        help='Max sample number.')
    parser.add_argument('--memory_bank', default='single', type=str,
                        choices=["multiple", "single"],
                        help='memory bank mode: "multiple", "single".')
    parser.add_argument('--rgb_backbone_name', default='vit_base_patch8_224_dino', type=str, 
                        choices=['vit_base_patch8_224_dino', 'vit_base_patch8_224', 'vit_base_patch8_224_in21k', 'vit_small_patch8_224_dino'],
                        help='Timm checkpoints name of RGB backbone.')
    parser.add_argument('--xyz_backbone_name', default='Point_MAE', type=str, choices=['Point_MAE', 'Point_Bert'],
                        help='Checkpoints name of RGB backbone[Point_MAE, Point_Bert].')
    parser.add_argument('--fusion_module_path', default='checkpoints/checkpoint-0.pth', type=str,
                        help='Checkpoints for fusion module.')
    parser.add_argument('--save_feature', default=False, action='store_true',
                        help='Save feature for training fusion block.')
    parser.add_argument('--use_uff', default=False, action='store_true',
                        help='Use UFF module.')
    parser.add_argument('--save_feature_path', default='datasets/patch_lib', type=str,
                        help='Save feature for training fusion block.')
    parser.add_argument('--save_preds', default=False, action='store_true',
                        help='Save predicts results.')
    parser.add_argument('--group_size', default=128, type=int,
                        help='Point group size of Point Transformer.')
    parser.add_argument('--num_group', default=16384, type=int,#16384
                        help='Point groups number of Point Transformer.')
    parser.add_argument('--random_state', default=None, type=int,
                        help='random_state for random project')
    parser.add_argument('--dataset_type', default='mvtec3d', type=str, choices=['mvtec3d', 'eyecandies'], 
                        help='Dataset type for training or testing')
    parser.add_argument('--dataset_path', default="/ssd2/m3lab/usrs/crt/DGCNN_core/3D-ADS/datasets/mvtec3d", type=str, 
                        help='Dataset store path')
    parser.add_argument('--img_size', default=224, type=int,
                        help='Images size for model')
    parser.add_argument('--xyz_s_lambda', default=1.0, type=float,
                        help='xyz_s_lambda')
    parser.add_argument('--xyz_smap_lambda', default=1.0, type=float,
                        help='xyz_smap_lambda')
    parser.add_argument('--rgb_s_lambda', default=0.1, type=float,
                        help='rgb_s_lambda')
    parser.add_argument('--rgb_smap_lambda', default=0.1, type=float,
                        help='rgb_smap_lambda')
    parser.add_argument('--fusion_s_lambda', default=1.0, type=float,
                        help='fusion_s_lambda')
    parser.add_argument('--fusion_smap_lambda', default=1.0, type=float,
                        help='fusion_smap_lambda')
    parser.add_argument('--coreset_eps', default=0.9, type=float,
                        help='eps for sparse project')
    parser.add_argument('--f_coreset', default=0.1, type=float,
                        help='eps for sparse project')
    parser.add_argument('--asy_memory_bank', default=None, type=int,
                        help='build an asymmetric memory bank for point clouds')
    parser.add_argument('--ocsvm_nu', default=0.5, type=float,
                        help='ocsvm nu')
    parser.add_argument('--ocsvm_maxiter', default=1000, type=int,
                        help='ocsvm maxiter')
    parser.add_argument('--rm_zero_for_project', default=False, action='store_true',
                        help='Save predicts results.')
    args = parser.parse_args()
    return args