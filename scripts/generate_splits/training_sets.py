import os 
import numpy as np 
import pandas as pd 
from sklearn.neighbors import KDTree 
import pickle 
import argparse 
from tqdm import tqdm 
from utils import TrainingTuple, load_csv, check_in_test_set
from utils import P1, P2, P3, P4, P5, P6
from utils import B1, B2, B3, B4, B5, B6, B7, B8, B9, B10, B11, B12
import matplotlib.pyplot as plt 
import open3d as o3d
from os.path import join
from scipy.spatial.transform import Rotation as R
import torch

# Test set boundaries 
POLY_VENMAN = [P1,P2,P3]
EXCLUDE_VENMAN = [B1, B2, B3, B4, B5, B6]

POLY_KARAWATHA = [P4,P5,P6]
EXCLUDE_KARAWATHA = [B7, B8, B9, B10, B11, B12]

_OFFSET = 2000

def find_overlapped_cloud(df, index1, index2):
    cloud1 = o3d.io.read_point_cloud(join(args.dataset_root,df.iloc[index1]["filename"]))
    cloud2 = o3d.io.read_point_cloud(join(args.dataset_root,df.iloc[index2]["filename"]))
    t1 = np.array([float(df['x'][index1]), float(df['y'][index1]), float(df['z'][index1])])
    q1 = np.array([float(df['qx'][index1]), float(df['qy'][index1]), float(df['qz'][index1]), float(df['qw'][index1])])
    r1 = R.from_quat(q1).as_matrix()
    T1 = np.eye(4)
    T1[:3,:3] = r1
    T1[:3,3] = t1
    cloud1.transform(T1)

    t2 = np.array([float(df['x'][index2]), float(df['y'][index2]), float(df['z'][index2])])
    q2 = np.array([float(df['qx'][index2]), float(df['qy'][index2]), float(df['qz'][index2]), float(df['qw'][index2])])
    r2 = R.from_quat(q2).as_matrix()
    T2 = np.eye(4)
    T2[:3,:3] = r2
    T2[:3,3] = t2
    cloud2.transform(T2)

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
    overlapped = cloud1.select_by_index(overlapped_cloud_indices)
    return len(overlapped.points)/len(cloud1.points)


def construct_query_dict(df_centroids, save_path, ind_nn_r, ind_r_r):
    # ind_nn_r: threshold for positive examples
    # ind_r_r: threshold for negative examples
    # Baseline dataset parameters in the original PointNetVLAD code: ind_nn_r=10, ind_r=50
    # Refined dataset parameters in the original PointNetVLAD code: ind_nn_r=12.5, ind_r=50
    tree = KDTree(df_centroids[['easting', 'northing']])
    ind_nn = tree.query_radius(df_centroids[['easting', 'northing']], r=ind_nn_r) # kdtree search
    ind_r = tree.query_radius(df_centroids[['easting', 'northing']], r=ind_r_r)
    queries = {} # dict, key是idx，item是traintuple

    for anchor_ndx in tqdm(range(len(ind_nn))):
        anchor_pos = np.array(df_centroids.iloc[anchor_ndx][['easting', 'northing']])
        pose = np.array(df_centroids.iloc[anchor_ndx][['x','y','z','qx','qy','qz','qw']])
        query = df_centroids.iloc[anchor_ndx]["filename"]
        # Extract timestamp from the filename
        scan_filename = os.path.split(query)[1]
        timestamp = float(os.path.splitext(scan_filename)[0])

        positives = ind_nn[anchor_ndx]
        non_negatives = ind_r[anchor_ndx] # opp is negative

        positives = positives[positives != anchor_ndx] 
        # Sort ascending order
        positives = np.sort(positives)
        positives_conf = np.zeros_like(positives,dtype=float)
        positives_dist = np.zeros_like(positives,dtype=float)

        non_negatives = np.sort(non_negatives)

        # Tuple(id: int, timestamp: str, rel_scan_filepath: str, positives: List[int], non_negatives: List[int])
        queries[anchor_ndx] = TrainingTuple(id=anchor_ndx, timestamp=timestamp, rel_scan_filepath=query,
                                            positives=positives, non_negatives=non_negatives, position=anchor_pos,
                                            positives_conf=positives_conf, positives_dist=positives_dist,
                                            pose = pose)

    file_path = save_path
    with open(file_path, 'wb') as handle:
        pickle.dump(queries, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Done ", save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate training dataset')
    parser.add_argument('--dataset_root', type = str, required = True, help = 'Dataset root folder')
    parser.add_argument('--save_folder', type = str, required = True, help = 'Folder to save training pickles to')
    parser.add_argument('--subset', type = str, required = True, help = 'v,k,vk')

    parser.add_argument('--csv_filename', type = str, default = 'poses_aligned.csv', help = 'Name of CSV containing ground truth poses')
    parser.add_argument('--cloud_folder', type = str, default = 'Clouds_downsampled', help = 'Name of folder containing point cloud frames')

    parser.add_argument('--pos_thresh', type = float, default = 5, help  = 'Threshold to sample positives within')
    parser.add_argument('--neg_thresh', type = float, default = 50, help = 'Threshold to sample negative within')

    parser.add_argument('--vis_splits', action = 'store_true', default = False, help = 'View training splits using matplotlib')
    args = parser.parse_args()

    assert os.path.exists(args.dataset_root), f"Cannot access dataset root folder: {args.dataset_root}"
    print(f'Dataset root: {args.dataset_root}')
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)

    ##### Venman #####
    venman_all_coords = []
    venman_all_c = []
    # Get training runs, i.e. first two sequence
    base = os.path.join(args.dataset_root, 'Venman')
    train_folders = sorted(os.listdir(base))[:2]

    df_train_venman = pd.DataFrame(columns=['filename', 'northing', 'easting', 'x','y','z','qx','qy','qz','qw' ])
    df_test_venman = pd.DataFrame(columns=['filename', 'northing', 'easting', 'x','y','z','qx','qy','qz','qw' ])
    if 'v' in args.subset:
        for idx, folder in enumerate(train_folders):
            df_locations = load_csv(os.path.join(base, folder, args.csv_filename), rel_cloud_path = os.path.join('Venman', folder, args.cloud_folder))

            for index, row in tqdm(df_locations.iterrows(), desc = f'Venman {idx+1} / {len(train_folders)}', total = len(df_locations)):
                venman_all_coords.append(row[['easting', 'northing']])
                row_split = check_in_test_set(row['easting'], row['northing'], test_polygons = POLY_VENMAN, exclude_polygons = EXCLUDE_VENMAN)
                if  row_split == 'test':
                    df_test_venman.loc[len(df_test_venman)] = row 
                    venman_all_c.append([1,0,0])
                elif row_split == 'train':
                    df_train_venman.loc[len(df_train_venman)] = row
                    venman_all_c.append([0,0,1])
                elif row_split == 'buffer':
                    venman_all_c.append([1, 165 / 255, 0])


    ##### Karawatha #####
    karawatha_all_coords = []
    karawatha_all_c = []
    # Get training runs, i.e. first two
    base = os.path.join(args.dataset_root, 'Karawatha')
    train_folders = sorted(os.listdir(base))[:2]

    df_train_karawatha = pd.DataFrame(columns=['filename', 'northing', 'easting', 'x','y','z','qx','qy','qz','qw' ])
    df_test_karawatha = pd.DataFrame(columns=['filename', 'northing', 'easting', 'x','y','z','qx','qy','qz','qw' ])
    if 'k' in args.subset:
        for idx, folder in enumerate(train_folders):
            df_locations = load_csv(os.path.join(base, folder, args.csv_filename), rel_cloud_path = os.path.join('Karawatha', folder, args.cloud_folder))

            for index, row in tqdm(df_locations.iterrows(), desc = f'Karawatha {idx+1} / {len(train_folders)}', total = len(df_locations)):
                karawatha_all_coords.append(row[['easting', 'northing']])
                row_split = check_in_test_set(row['easting'], row['northing'], test_polygons = POLY_KARAWATHA, exclude_polygons = EXCLUDE_KARAWATHA)
                if  row_split == 'test':
                    df_test_karawatha.loc[len(df_test_karawatha)] = row 
                    karawatha_all_c.append([1,0,0])
                elif row_split == 'train':
                    df_train_karawatha.loc[len(df_train_karawatha)] = row
                    karawatha_all_c.append([0,0,1])
                elif row_split == 'buffer':
                    karawatha_all_c.append([1, 165 / 255, 0])

        # Offset Karawatha by a large number to stop the ground truth maps from overlapping
        df_train_karawatha[['x', 'easting']] = df_train_karawatha[['x', 'easting']] + _OFFSET
        df_test_karawatha[['x', 'easting']] = df_test_karawatha[['x', 'easting']] + _OFFSET
        # karawatha_all_coords = [[x[0] + _OFFSET, x[1]] for x in karawatha_all_coords]

        #### Combined ####
    df_train_both = pd.concat([df_train_venman, df_train_karawatha], ignore_index = True)
    df_test_both = pd.concat([df_test_venman, df_test_karawatha], ignore_index = True)

    all_coords = np.array(venman_all_coords + karawatha_all_coords)
    all_c = np.array(venman_all_c + karawatha_all_c)


    ### Summary ###
    if 'v' in args.subset:
        print('\nVenman: ')
        total_submaps = len(venman_all_coords)
        print(f"\tNumber of training submaps: {len(df_train_venman)} ({len(df_train_venman) / total_submaps})%")
        print(f"\tNumber of non-disjoint test submaps: {len(df_test_venman)} ({len(df_test_venman) / total_submaps})%")
        print(f"\tNumber of excluded submaps in bufferzone {total_submaps - len(df_train_venman) - len(df_test_venman)} ({(total_submaps - len(df_train_venman) - len(df_test_venman)) * 100 / total_submaps}%)")

    if 'k' in args.subset:
        print('\nKarawatha: ')
        total_submaps = len(karawatha_all_coords)
        print(f"\tNumber of training submaps: {len(df_train_karawatha)} ({len(df_train_karawatha) / total_submaps})%")
        print(f"\tNumber of non-disjoint test submaps: {len(df_test_karawatha)} ({len(df_test_karawatha) / total_submaps})%")
        print(f"\tNumber of excluded submaps in bufferzone {total_submaps - len(df_train_karawatha) - len(df_test_karawatha)} ({(total_submaps - len(df_train_karawatha) - len(df_test_karawatha)) * 100 / total_submaps}%)")

    if 'vk' in args.subset:
        print('\nBoth: ')
        total_submaps = len(all_coords)
        print(f"\tNumber of training submaps: {len(df_train_both)} ({len(df_train_both) / total_submaps})%")
        print(f"\tNumber of non-disjoint test submaps: {len(df_test_both)} ({len(df_test_both) / total_submaps})%")
        print(f"\tNumber of excluded submaps in bufferzone {total_submaps - len(df_train_both) - len(df_test_both)} ({(total_submaps - len(df_train_both) - len(df_test_both)) * 100 / total_submaps}%)")


    ### Vis if selected ###
    if args.vis_splits == True:
        venman_all_coords = np.array(venman_all_coords)
        karawatha_all_coords = np.array(karawatha_all_coords)

        fig, (ax2, ax3) = plt.subplots(1, 2)
        # ax1.scatter(all_coords[:,0], all_coords[:,1], c = all_c)

        ax2.scatter(venman_all_coords[:,0], venman_all_coords[:,1], c = venman_all_c)
        for poly in POLY_VENMAN:
            ax2.plot(*poly.exterior.xy)

        ax3.scatter(karawatha_all_coords[:,0], karawatha_all_coords[:,1], c = karawatha_all_c)
        for poly in POLY_KARAWATHA:
            ax3.plot(*poly.exterior.xy)
        plt.show()

    construct_query_dict(df_train_both, os.path.join(args.save_folder, f"1_training_wild-places.pickle"), ind_nn_r = args.pos_thresh, ind_r_r = args.neg_thresh)
    construct_query_dict(df_test_both, os.path.join(args.save_folder, f"1_testing_wild-places.pickle"), ind_nn_r = args.pos_thresh, ind_r_r = args.neg_thresh)