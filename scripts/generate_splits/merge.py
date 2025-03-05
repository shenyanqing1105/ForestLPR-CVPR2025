import numpy as np
# python -m pip install numpy-quaternion
import pandas as pd
from scipy.spatial.transform import Rotation as R
import open3d as o3d
from os.path import join
import sys

root_path = '/A_dataset/'
out_path = '/Dataset'
df = pd.read_csv(join(root_path,'poses_aligned.csv'), delimiter = ',', dtype = str)
folder = 'Clouds_downsampled'# 'Clouds_downsampled', Clouds_normalized

def down():
    N = len(df['timestamp'])
    for i in range(N):
        pcd = o3d.io.read_point_cloud(join(root_path, 'Clouds', df['timestamp'][i]+'.pcd'))
        downpcd = pcd.voxel_down_sample(voxel_size=0.1)
        # o3d.visualization.draw_geometries([downpcd])
        o3d.io.write_point_cloud(join(out_path, 'Clouds_downsampled', df['timestamp'][i]+'.pcd'), downpcd)
        if i%100 == 0:
            print('processing {}/{}'.format(i, N))

def filter():
    N = len(df['timestamp'])
    for i in range(N):
        pcd = o3d.io.read_point_cloud(join(root_path, 'Clouds_raw', df['timestamp'][i]+'.pcd'))
        points = np.asarray(pcd.points)
        filtered_points = []
        for point in points:
            x, y, z = point
            if np.sqrt(x**2 + y**2) > 2:
                filtered_points.append([x, y, z])
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(filtered_points)
        o3d.io.write_point_cloud(join(root_path, 'Clouds_downsampled', df['timestamp'][i]+'.pcd'), pcd)
        if i%100 == 0:
            print('processing {}/{}'.format(i, N))


def quaternion_to_rotation_matrix(x,y,z,w):
    """
    Convert a quaternion into a 3x3 rotation matrix.

    :param quaternion: A list or array with 4 elements [x, y, z, w]
    :return: A 3x3 rotation matrix
    """
    r00 = 1 - 2 * (y**2 + z**2)
    r01 = 2 * (x*y - z*w)
    r02 = 2 * (x*z + y*w)
    r10 = 2 * (x*y + z*w)
    r11 = 1 - 2 * (x**2 + z**2)
    r12 = 2 * (y*z - x*w)
    r20 = 2 * (x*z - y*w)
    r21 = 2 * (y*z + x*w)
    r22 = 1 - 2 * (x**2 + y**2)

    rotation_matrix = np.array([
        [r00, r01, r02],
        [r10, r11, r12],
        [r20, r21, r22]
    ])

    return rotation_matrix

def merge(flag_rotate='False', flag_filter='False'):
    # load pose_aligned.csv
    N = 5 # 11 frames
    M = 5 # every 5 frame as the key frame
    sub = len(df['timestamp']) // M

    for i in range(sub-2):
        index_ref = N+i*M

        t_ref = np.array([float(df['x'][index_ref]), float(df['y'][index_ref]), 0])
        T_ref = np.eye(4)
        T_ref[:3,3] = t_ref
        fuse_pcd = o3d.geometry.PointCloud()
        for j in range(2*N+1):
            index_lidar = i*M + j
            pcd = o3d.io.read_point_cloud(join(root_path, folder, df['timestamp'][index_lidar]+'.pcd'))
            if flag_filter=='True':

                points = np.asarray(pcd.points)

                filtered_points = []
                for point in points:
                    x, y, z = point
                    if np.sqrt(x**2 + y**2) > 1.5:
                        filtered_points.append([x, y, z])

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(filtered_points)
            T_lidar = np.eye(4)
            t_lidar = np.array([float(df['x'][index_lidar]), float(df['y'][index_lidar]), float(df['z'][index_lidar])])
            T_lidar[:3,3] = t_lidar
            if flag_rotate=='True':

                q_lidar = [float(df['qx'][index_lidar]), float(df['qy'][index_lidar]), float(df['qz'][index_lidar]), float(df['qw'][index_lidar])]
                r_lidar = R.from_quat(q_lidar).as_matrix()
                T_lidar[:3,:3] =  r_lidar
            pcd.transform(T_lidar)

            fuse_pcd += pcd
        T = np.linalg.inv(T_ref) # back to zero
        fuse_pcd.transform(T)

        lidar = np.asarray(fuse_pcd.points)

        TOP_Y_MIN = -30
        TOP_Y_MAX = +30
        TOP_X_MIN = -30
        TOP_X_MAX = 30


        idx = np.where (lidar[:,0]>TOP_X_MIN)
        lidar = lidar[idx]
        idx = np.where (lidar[:,0]<TOP_X_MAX)
        lidar = lidar[idx]

        idx = np.where (lidar[:,1]>TOP_Y_MIN)
        lidar = lidar[idx]
        idx = np.where (lidar[:,1]<TOP_Y_MAX)
        lidar = lidar[idx]

        fuse_pcd = o3d.geometry.PointCloud()
        fuse_pcd.points = o3d.utility.Vector3dVector(lidar)
        o3d.io.write_point_cloud(join(out_path, folder, df['timestamp'][index_ref]+'.pcd'), fuse_pcd)


        if i%100 == 0:
            print('processing {}/{}'.format(i, sub))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python merge.py <IF_ROTATE> <IF_FILTER>")
        sys.exit(1)
    flag_rotate = sys.argv[1]
    flag_filter = sys.argv[2]
    # merge(flag_rotate, flag_filter)
    # down()
    filter()