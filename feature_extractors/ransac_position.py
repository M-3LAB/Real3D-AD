import numpy as np
import random
import open3d as o3d
import copy
# import sys
# sys.path.append('../')
# from utils.visualization import vis_pointcloud_np_two
def vis_pointcloud_np_two(xyz=None,xyz2=None):
    """
    xyz = numpy.array N*3
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(xyz2)
    o3d.visualization.draw_geometries([pcd,pcd2])

def norm_pcd(pcd):
    points_coord = np.asarray(pcd.points)
    # print(points_coord.shape)
    center = np.average(points_coord,axis=0)
    # print(center.shape)
    new_points = points_coord-np.expand_dims(center,axis=0)
    pcd.points = o3d.utility.Vector3dVector(new_points)
    return pcd

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4559,
                                      front=[0.6452, -0.3036, -0.7011],
                                      lookat=[1.9892, 2.0208, 1.8945],
                                      up=[-0.2779, -0.9482, 0.1556])

def preprocess_point_cloud(pcd, voxel_size):
    # print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    # print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    # print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def prepare_dataset(voxel_size,source_data,target_data):
    # print(":: Load two point clouds and disturb initial pose.")

    # demo_icp_pcds = o3d.data.DemoICPPointClouds()
    # source = o3d.io.read_point_cloud(demo_icp_pcds.paths[0])
    # target = o3d.io.read_point_cloud(demo_icp_pcds.paths[1])
    # source = o3d.io.read_point_cloud('/ssd2/m3lab/data/3DAD/3dad_demo_more_pcd/airplane/60_template.pcd')
    # target = o3d.io.read_point_cloud('/ssd2/m3lab/data/3DAD/3dad_demo_more_pcd/airplane/67_good.pcd')
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(source_data)
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(target_data)
    # source = o3d.io.read_point_cloud('/ssd2/m3lab/data/3DAD/3dad_demo_more_pcd/barCandy/45_template.pcd')
    # target = o3d.io.read_point_cloud('/ssd2/m3lab/data/3DAD/3dad_demo_more_pcd/barCandy/396_bulge.pcd')
    
    # source = norm_pcd(source)
    # target = norm_pcd(target)
    
    trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    source.transform(trans_init)
    # draw_registration_result(source, target, np.identity(4))

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh



def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    # print(":: RANSAC registration on downsampled point clouds.")
    # print("   Since the downsampling voxel size is %.3f," % voxel_size)
    # print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result

def get_registration_np(source_data, target_data):
    voxel_size = 0.5  # means 5cm for this dataset
    source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(
        voxel_size,source_data,target_data)
    
    # source = o3d.geometry.PointCloud()
    # source.points = o3d.utility.Vector3dVector(source_data)
    result_ransac = execute_global_registration(source_down, target_down,
                                                source_fpfh, target_fpfh,
                                                voxel_size)
    # print(result_ransac)
    # draw_registration_result(source_down, target_down, result_ransac.transformation)

    # source = o3d.io.read_point_cloud('/ssd2/m3lab/data/3DAD/3dad_demo_more_pcd/airplane/60_template.pcd')
    # target = o3d.io.read_point_cloud('/ssd2/m3lab/data/3DAD/3dad_demo_more_pcd/airplane/67_good.pcd')
    # target = o3d.geometry.PointCloud()
    # target.points = o3d.utility.Vector3dVector(target_data)
    
    # source = norm_pcd(source)
    # target = norm_pcd(target)
    
    # trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
    #                             [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    source.transform(result_ransac.transformation)
    # draw_registration_result(source, target, result_ransac.transformation)
    return np.asarray(source.points)

def get_registration_refine_np(source_data, target_data):
    voxel_size = 0.5  # means 5cm for this dataset
    source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(
        voxel_size,source_data,target_data)
    result_ransac = execute_global_registration(source_down, target_down,
                                                source_fpfh, target_fpfh,
                                                voxel_size)
    source.transform(result_ransac.transformation)
    distance_threshold = voxel_size * 0.4
    result = o3d.pipelines.registration.registration_icp(source, target, 0.1)
    source.transform(result.transformation)
    return np.asarray(source.points)
    
# source = o3d.io.read_point_cloud('/ssd2/m3lab/data/3DAD/3dad_demo_more_pcd/barCandy/45_template.pcd')
# target = o3d.io.read_point_cloud('/ssd2/m3lab/data/3DAD/3dad_demo_more_pcd/barCandy/396_bulge.pcd')
# source = norm_pcd(source)
# target = norm_pcd(target)
# trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
#                              [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
# source.transform(trans_init)
# source_data = np.asarray(source.points)
# target_data = np.asarray(target.points)

# source_data = get_registration_np(source_data,target_data)

# # source.points = o3d.utility.Vector3dVector(source_data)
# vis_pointcloud_np_two(source_data,target_data)