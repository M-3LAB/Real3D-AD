from utils.mvtec3d_util import *
import open3d as o3d
import numpy as np
import torch
from feature_extractors.features import Features


def get_fpfh_features(organized_pc, voxel_size=0.05):
    organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
    unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
    # print(unorganized_pc.shape) 50176*3
    nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
    unorganized_pc_no_zeros = unorganized_pc[nonzero_indices, :]
    # print(unorganized_pc_no_zeros.shape) #42338*3
    o3d_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(unorganized_pc_no_zeros))

    radius_normal = voxel_size * 2
    o3d_pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(o3d_pc, o3d.geometry.KDTreeSearchParamHybrid
    (radius=radius_feature, max_nn=100))
    fpfh = pcd_fpfh.data.T
    # print(fpfh.shape) 42175,41961,42340,42117,42031 33
    full_fpfh = np.zeros((unorganized_pc.shape[0], fpfh.shape[1]), dtype=fpfh.dtype)
    full_fpfh[nonzero_indices, :] = fpfh
    full_fpfh_reshaped = full_fpfh.reshape((organized_pc_np.shape[0], organized_pc_np.shape[1], fpfh.shape[1]))
    full_fpfh_tensor = torch.tensor(full_fpfh_reshaped).permute(2, 0, 1).unsqueeze(dim=0)
    return full_fpfh_tensor


class FPFHFeatures(Features):
    def add_sample_to_mem_bank(self, sample):
        fpfh_feature_maps = get_fpfh_features(sample[1])
        # print(fpfh_feature_maps.shape)1 33 224 224
        fpfh_feature_maps_resized = self.resize(self.average(fpfh_feature_maps))
        fpfh_patch = fpfh_feature_maps_resized.reshape(fpfh_feature_maps_resized.shape[1], -1).T
        # print(fpfh_patch.shape) torch.Size([784, 33])
        self.patch_lib.append(fpfh_patch)

    def predict(self, sample, mask, label):
        # print(type(sample))list
        # print(type(sample[0]))tensor
        # print(sample[0].shape) 1 3 224 224
        # print(type(mask)) tensor
        # print(mask.shape)1 1 224 224
        # print(type(label)) tensor
        depth_feature_maps = get_fpfh_features(sample[1])
        depth_feature_maps_resized = self.resize(self.average(depth_feature_maps))
        patch = depth_feature_maps_resized.reshape(depth_feature_maps_resized.shape[1], -1).T

        # print(type(depth_feature_maps_resized)) tensor
        # print(type(patch))tensor
        # print(patch.shape) 784 33
        self.compute_s_s_map(patch, depth_feature_maps_resized.shape[-2:], mask, label)
