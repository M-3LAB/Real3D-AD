import open3d as o3d
from dataset_pc import Dataset3dad_train,Dataset3dad_test
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from feature_extractors.fpfh_pc_features import PC_FPFHFeatures
import torch
import torch.nn.functional as F
from feature_extractors.ransac_position import get_registration_np,get_registration_refine_np
from utils.visualization import vis_pointcloud_np_two
import os

real_3d_classes = ['airplane','car','candybar','chicken',
                   'diamond','duck','fish','gemstone',
                   'seahorse','shell','starfish','toffees']

# data_dir = "/ssd2/m3lab/data/3DAD/3dad_demo_pcd_tiny/"
root_dir = './data'
save_dir = './benchmark/btf_raw/'
# data_dir = root_dir + 'airplane'
print('Task start: BTF_Raw')
for real_class in real_3d_classes:
    train_loader = DataLoader(Dataset3dad_train(root_dir, real_class, 1024, True), num_workers=1,
                                batch_size=1, shuffle=False, drop_last=True)

    voxel_size = 0.5
    train_sampling_ratio = 100
    test_sampling_ratio = 500
    pc_pfph = PC_FPFHFeatures()
    for data, mask, label, path in train_loader:
        # print(path)
        # path_list = path[0].split('/')
        # print(path_list)
        basic_template = data.squeeze(0).cpu().numpy()
        break
    
     
    for data, mask, label, path in train_loader:
        reg_data = data.squeeze(0).cpu().numpy()
        # reg_data = get_registration_refine_np(data.squeeze(0).cpu().numpy(),basic_template)
        
        # vis_pointcloud_np_two(reg_data,basic_template)
        # print(data.shape) torch.Size([1, 190339, 3])
        # o3d_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(data.squeeze(0).numpy()))
        # radius_normal = voxel_size * 2
        # o3d_pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    
        # radius_feature = voxel_size * 5
        # pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(o3d_pc, o3d.geometry.KDTreeSearchParamHybrid
        # (radius=radius_feature, max_nn=100))
        # fpfh = torch.from_numpy(pcd_fpfh.data.T)
        fpfh = torch.from_numpy(reg_data)
        length = fpfh.shape[0]
        idx_list = [i*train_sampling_ratio for i in range(int(length/train_sampling_ratio))]
        new_fpfh = fpfh[idx_list,:]
        # pc_pfph.patch_lib.append(fpfh)
        pc_pfph.patch_lib.append(new_fpfh)
        # print(fpfh.shape) (190339, 33)
        
    #1. test datalodaer 2. predict anomaly map 3. metrics point_cloud_auc point_cloud_ap
        # print(data)
        # print(label)

    # Pipeline 1. 存储train data feature到memory 2.抽取test data feature，与meomory中feature比较 3.对每个点计算metrics

    test_loader = DataLoader(Dataset3dad_test(root_dir, real_class, 1024, True), num_workers=1,
                                batch_size=1, shuffle=True, drop_last=False)
    if( not os.path.exists(save_dir+real_class)):
        os.makedirs(save_dir+real_class)
    for data, mask, label, path in test_loader:
        reg_data = data.squeeze(0).cpu().numpy()
        # reg_data = get_registration_refine_np(data.squeeze(0).cpu().numpy(),basic_template)
        
        # vis_pointcloud_np_two(reg_data,basic_template)
        
        # print('data shape:{}'.format(str(data.shape)))
        # o3d_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(data.squeeze(0).numpy()))
        # radius_normal = voxel_size * 2
        # o3d_pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

        # radius_feature = voxel_size * 5
        # pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(o3d_pc, o3d.geometry.KDTreeSearchParamHybrid
        # (radius=radius_feature, max_nn=100))
        # fpfh = pcd_fpfh.data.T
        # fpfh = torch.from_numpy(pcd_fpfh.data.T)
        fpfh = torch.from_numpy(reg_data)
        length = fpfh.shape[0]
        idx_list = [i*test_sampling_ratio for i in range(int(length/test_sampling_ratio))]
        new_fpfh = fpfh[idx_list,:]
        # new_mask = mask[:,idx_list]
        path_list = path[0].split('/')
        target_path = os.path.join(save_dir,real_class,path_list[-1])
        pc_pfph.compute_s_s_map(new_fpfh,33,mask,label,reg_data,idx_list,path[0],target_path)
        # pass
        # print(data.shape) torch.Size([1, 366855, 3])
        # print(label.shape) torch.Size([1, 366855])
    pc_pfph.calculate_metrics()
    image_rocaucs = round(pc_pfph.image_rocauc, 3)
    image_aupr = round(pc_pfph.image_aupr, 3)
    pixel_rocaucs = round(pc_pfph.pixel_rocauc, 3)
    pixel_aupr = round(pc_pfph.pixel_aupr, 3)
    # au_pros = round(pc_pfph.au_pro, 3)
    # print(real_class)
    # print('image_rocaucs:'+str(image_rocaucs))
    # print('image_aupr:'+str(image_aupr))
    # print('pixel_rocaucs:'+str(pixel_rocaucs))
    # print('pixel_aupr:'+str(pixel_aupr))
    print('Task:{}, object_auroc:{}, point_auroc:{}, object_aupr:{}, point_aupr:{}'.format
                    (cls,image_rocaucs,pixel_rocaucs,image_aupr,pixel_aupr))