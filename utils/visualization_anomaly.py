import open3d as o3d
import numpy as np
import cv2

def vis_pointcloud(path=None):
    """
    path: pcd file path
    """
    pcd = o3d.io.read_point_cloud(path)
    # print(type(pcd))<class 'open3d.cuda.pybind.geometry.PointCloud'>
    # print(type(pcd.points))<class 'open3d.cuda.pybind.utility.Vector3dVector'>
    # xyzrgb = np.array(pcd.points)
    # print(xyzrgb.shape) N*3
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    o3d.visualization.draw_geometries([pcd])
    
def vis_pointcloud_withcoord(path=None):
    """
    path: pcd file path
    """
    pcd = o3d.io.read_point_cloud(path)
    FOR1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=15, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([pcd,FOR1],
                                    # zoom=0.5,
                                    # front=[0,0,1],
                                    # lookat=[0,1,0],
                                    # up=[1,0,0]
                                    # front=[0.4257, -0.2125, -0.8795],
                                    # lookat=[2.6172, 2.0475, 1.532],
                                    # up=[-0.0694, -0.9768, 0.2024]
                                    )

def vis_pointcloud_np(xyz=None):
    """
    xyz = numpy.array N*3
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    o3d.visualization.draw_geometries([pcd])
    
def vis_pointcloud_np_two(xyz=None,xyz2=None):
    """
    xyz = numpy.array N*3
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(xyz2)
    o3d.visualization.draw_geometries([pcd,pcd2])

def vis_pointcloud_gt(path=None):
    gt_pc = np.loadtxt(path)
    # print(gt_pc.shape)
    gt = gt_pc[:,3]
    # colors_pc = np.tile(new_pc, (1, 3))
    new_colors = np.zeros_like(gt_pc[:,:3])
    anomaly_pos = gt==1
    normal_pos = gt==0
    new_colors[normal_pos] = [0,0,1]
    new_colors[anomaly_pos] = [1,0,0]
    # new_colors[colors_pc==1] = [0.2,0.5,0.7]
    # # print(new_matrix.shape)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(gt_pc[:,:3])
    pcd.colors = o3d.utility.Vector3dVector(new_colors)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    o3d.visualization.draw_geometries([pcd])
    # o3d.visualization.draw_geometries([pcd], zoom=0.69999999999999996,
    #                                 front=[0.46855077365287628, -0.50210647509955675, -0.72687637200034871],
    #                                 lookat=[107.94386673, -6.4175922850000013, 226.26735687500002],
    #                                 up=[-0.45777569937090423, 0.56571248541038721, -0.68586499612991059],
    #                                 )
    
def vis_pointcloud_gt_voxel_down(path=None,voxel_size=0.5):
    gt_pc = np.loadtxt(path)
    # print(gt_pc.shape)
    gt = gt_pc[:,3]
    # colors_pc = np.tile(new_pc, (1, 3))
    new_colors = np.zeros_like(gt_pc[:,:3])
    anomaly_pos = gt==1
    normal_pos = gt==0
    new_colors[normal_pos] = [0.4,0.4,0.4]
    new_colors[anomaly_pos] = [1,0,0]
    # new_colors[colors_pc==1] = [0.2,0.5,0.7]
    # # print(new_matrix.shape)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(gt_pc[:,:3])
    pcd.colors = o3d.utility.Vector3dVector(new_colors)
    pcd_new = o3d.geometry.PointCloud.voxel_down_sample(pcd, voxel_size)
    pcd_new.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    o3d.visualization.draw_geometries([pcd_new])

def cv2heatmap(gray):
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap

def vis_pointcloud_anomalymap(point_cloud, anomaly_map):
    # point_cloud numpy,(n)
    # anomaly 0-1 float
    heatmap = cv2heatmap(anomaly_map*255)/255
    heatmap = heatmap.squeeze()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    pcd.colors = o3d.utility.Vector3dVector(heatmap)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    o3d.visualization.draw_geometries([pcd])
    
def vis_pointcloud_anomalymap_pcdpath(pcd_path, anomaly_map):
    # point_cloud numpy,(n)
    # anomaly 0-1 float
    heatmap = cv2heatmap(anomaly_map*255)/255
    heatmap = heatmap.squeeze()
    pcd = o3d.io.read_point_cloud(pcd_path)
    pcd.colors = o3d.utility.Vector3dVector(heatmap)
    o3d.visualization.draw_geometries([pcd])

def save_anomalymap(pcd_path, anomaly_map, target_path):
    #pad_path pure point cloud path
    #anomaly map numpy(n),range(0-1)
    #target_path save path
    heatmap = cv2heatmap(anomaly_map*255)/255
    heatmap = heatmap.squeeze()
    pcd = o3d.io.read_point_cloud(pcd_path)
    pcd.colors = o3d.utility.Vector3dVector(heatmap)
    o3d.io.write_point_cloud(target_path, pcd)
    
def norm_pcd(pcd):
    points_coord = np.asarray(pcd.points)
    # print(points_coord.shape)
    center = np.average(points_coord,axis=0)
    # print(center.shape)
    new_points = points_coord-np.expand_dims(center,axis=0)
    pcd.points = o3d.utility.Vector3dVector(new_points)
    return pcd

def norm_numpy(anomaly):
    min_ano = np.min(anomaly)
    max_ano = np.max(anomaly)
    anomaly = (anomaly-min_ano)/(max_ano-min_ano)
    return anomaly

def down_sample_voxel(pcd,voxel_size):
    pcd_new = o3d.geometry.PointCloud.voxel_down_sample(pcd, voxel_size)
    return pcd_new

def load_view_point(pcd, filename):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='pcd', width=1440, height=1080)
    ctr = vis.get_view_control()
    param = o3d.io.read_pinhole_camera_parameters(filename)
    vis.add_geometry(pcd)
    vis.get_render_option().load_from_json('renderoption.json')
    ctr.convert_from_pinhole_camera_parameters(param)
    vis.run()
    vis.destroy_window()