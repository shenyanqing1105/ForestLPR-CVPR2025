# coding=utf-8
'''
Author: shenyanqing1105 1159364090@qq.com
Date: 2024-11-21 02:59:53
LastEditors: shenyanqing1105 1159364090@qq.com
LastEditTime: 2025-03-05 16:00:37
FilePath: /ForestLPR-CVPR2025/scripts/overlap.py
'''
import open3d as o3d
from os.path import join
import pandas as pd
from scipy.spatial.transform import Rotation
import numpy as np
from scipy.spatial.transform import Rotation as R

root_dir = '/A_dataset/Wild-Places/Venman/V-04/'
df = pd.read_csv(join(root_dir,'poses_aligned.csv'), delimiter = ',', dtype = str)

def find_overlapped_cloud(cloud1, cloud2):
    overlapped_cloud_indices = []
    octree = o3d.geometry.Octree(max_depth=4)
    octree.convert_from_point_cloud(cloud1, size_expand=0.01)
    min_pt = octree.get_min_bound()
    max_pt = octree.get_max_bound()
    for point in cloud2.points:
        if point[0] < min_pt[0] or point[1] < min_pt[1] or point[2] < min_pt[2] or \
                point[0] > max_pt[0] or point[1] > max_pt[1] or point[2] > max_pt[2]:
            continue
        else:
            leaf_node, leaf_info = octree.locate_leaf_node(point)
            if leaf_info is not None:
                indices = leaf_node.indices
                for indice in indices:
                    overlapped_cloud_indices.append(indice)
    # o3d.visualization.draw_geometries([octree])
    return cloud1.select_by_index(overlapped_cloud_indices)


if __name__ == '__main__':
    # "read cloud"
    # (x, y, z, w) 格式 !!!!!!!!
    for index_ref in range(len(df['timestamp'])):

        pcd_ref = o3d.io.read_point_cloud(join(root_dir, 'Clouds_downsampled', df['timestamp'][index_ref]+'.pcd'))
        q_ref = np.array([float(df['qx'][index_ref]), float(df['qy'][index_ref]), float(df['qz'][index_ref]), float(df['qw'][index_ref])])
        r_ref = Rotation.from_quat(q_ref).as_matrix()
        t_ref = np.array([float(df['x'][index_ref]), float(df['y'][index_ref]), float(df['z'][index_ref])])
        T_ref = np.eye(4)
        T_ref[:3,:3] = r_ref
        T_ref[:3,3] = t_ref
        pcd_ref.transform(T_ref)
        for index_lidar in range(200, len(df['timestamp'])):
            pcd_lidar = o3d.io.read_point_cloud(join(root_dir, 'Clouds_downsampled', df['timestamp'][index_lidar]+'.pcd'))
            q_lidar = np.array([float(df['qx'][index_lidar]), float(df['qy'][index_lidar]), float(df['qz'][index_lidar]), float(df['qw'][index_lidar])])
            r_lidar = Rotation.from_quat(q_lidar).as_matrix()
            t_lidar = np.array([float(df['x'][index_lidar]), float(df['y'][index_lidar]), float(df['z'][index_lidar])])
            T_lidar = np.eye(4)
            T_lidar[:3,:3] = r_lidar
            T_lidar[:3,3] = t_lidar
            r = R.from_matrix(r_lidar)
            euler_angles = r.as_euler('xyz', degrees=True)
            r1 = R.from_matrix(r_ref)
            euler_angles1 = r1.as_euler('xyz', degrees=True)
            pcd_lidar.transform(T_lidar)

    # "find overlapped cloud"
            overlapped_cloud2 = find_overlapped_cloud(pcd_lidar, pcd_ref)
            overlapped_cloud1 = find_overlapped_cloud(pcd_ref, pcd_lidar)
            print(len(pcd_ref.points))
            print(len(pcd_lidar.points))
            print(len(overlapped_cloud1.points))
            print(len(overlapped_cloud2.points))
            o3d.io.write_point_cloud(join(root_dir, df['timestamp'][index_lidar]+'.pcd'), pcd_lidar)
            o3d.io.write_point_cloud(join(root_dir, df['timestamp'][index_ref]+'.pcd'), pcd_ref)


    
    # "save cloud"
    # o3d.io.write_point_cloud("D:/o3d_overlapped1.pcd", overlapped_cloud1)
    # o3d.io.write_point_cloud("D:/o3d_overlapped2.pcd", overlapped_cloud2)

