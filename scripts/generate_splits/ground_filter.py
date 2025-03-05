# coding: utf-8
# import laspy
import open3d as o3d
import CSF
import numpy as np
import pandas as pd
from os.path import join
from scipy.spatial import KDTree

root_path = '/A_dataset/Wild-Places/Venman/V-03' # Karawatha
out_path = '/A_dataset/'
df = pd.read_csv(join(root_path,'poses_aligned.csv'), delimiter = ',', dtype = str)
FLAG_FILTER = True

def extract_ground(csf, id): # CSF method, using logg3d env
    pcd = o3d.io.read_point_cloud(join(root_path, 'Clouds_downsampled', df['timestamp'][id]+'.pcd'))

    xyz = np.array(pcd.points)

    csf.setPointCloud(xyz)
    ground = CSF.VecInt()  # a list to indicate the index of ground points after calculation, initialize
    non_ground = CSF.VecInt() # a list to indicate the index of non-ground points after calculation, initialize
    csf.do_filtering(ground, non_ground) # do actual filtering. core of algorithm
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(xyz[np.array(non_ground)])
    non_ground = np.asarray(point_cloud.points)
    # o3d.visualization.draw_geometries([point_cloud])
    o3d.io.write_point_cloud(join(out_path, 'Clouds_noground', df['timestamp'][id]+'.pcd'), point_cloud)

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(xyz[np.array(ground)])
    ground = np.asarray(point_cloud.points)
    # o3d.visualization.draw_geometries([point_cloud])
    o3d.io.write_point_cloud(join(out_path, 'Clouds_ground', df['timestamp'][id]+'.pcd'), point_cloud)


    # height offset removal
    tree = KDTree(ground[:,:2])
    radius = 3
    for idx, point in enumerate(non_ground):
        indices = tree.query_ball_point(point[:2], radius)
        rr = radius
        while len(indices) == 0:
            rr += 1
            indices = tree.query_ball_point(point[:2], rr)
        if len(indices) > 0:
            subz_N = 0
            subz_D = 0
            for i in indices:
                diff = point[:2] - ground[i][:2]
                l2 = np.sum(diff ** 2)
                subz_N += 1/l2 * ground[i][2]
                subz_D += 1/l2
            subz = subz_N/subz_D
            non_ground[idx,2] -= subz


    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(non_ground)
    # o3d.visualization.draw_geometries([point_cloud])
    o3d.io.write_point_cloud(join(out_path, 'Clouds_normalized', df['timestamp'][id]+'.pcd'), point_cloud)




if __name__ == "__main__":
    N = len(df['timestamp'])

    csf = CSF.CSF()

    # prameter settings
    csf.params.bSloopSmooth = True
    csf.params.cloth_resolution = 0.3
    csf.params.rigidness = 3
    csf.params.time_step = 0.65
    csf.params.class_threshold = 0.5
    csf.params.interactions = 500
    # more details about parameter: http://ramm.bnu.edu.cn/projects/CSF/download/

    if 0:
        # for demo
        csf.readPointsFromFile('../../sample.txt')
        ground = CSF.VecInt()  # a list to indicate the index of ground points after calculation, initialize
        non_ground = CSF.VecInt() # a list to indicate the index of non-ground points after calculation, initialize
        csf.do_filtering(ground, non_ground) # do actual filtering. core of algorithm
        with open('../../sample.txt', 'r') as file:
            # 按行读取文件内容
            matrix = []
            for line in file:
                # 使用split()方法按空格拆分每行内容，并将结果转换为整数类型
                row = [x for x in line.split()]
                # 将拆分后的行添加到二维数组中
                matrix.append(row)
        xyz = np.asarray(matrix)
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(xyz[np.array(ground)])
        non_ground = np.asarray(point_cloud.points)
        o3d.visualization.draw_geometries([point_cloud])


    for i in range(N):

        extract_ground(csf, i)

        if i%100 == 0:
            print('-----------------processing {}/{}-------------------'.format(i, N))