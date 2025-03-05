# coding=utf-8
'''
Author: shenyanqing
Description: 
'''
from tqdm import tqdm 
import torch 
import numpy as np 
import argparse 
import os 
import pickle 
import pandas as pd
from torchpack.utils.config import configs 
from mc_utils import * 
from collections import Counter
from collections import deque
import cv2


# ORB config
nfeatures = 500
scale_factor = 1.00
n_levels = 1
first_level = 0
WTA_K = 2
edge_threshold = 31
score_type = cv2.ORB_HARRIS_SCORE if 0 == 0 else cv2.ORB_FAST_SCORE
patch_size = 31
fast_threshold = 35
# BEV config
TOP_Z_MIN = 2
TOP_Z_MAX = 6
MODE='density' # density, height
PROCESS='normalized' # normalized, original
HEIGHT=True # if crop or not

orb = cv2.ORB_create(
    nfeatures=nfeatures,
    scaleFactor=scale_factor,
    nlevels=n_levels,
    edgeThreshold=edge_threshold,
    firstLevel=first_level,
    WTA_K=WTA_K,
    scoreType=score_type,
    patchSize=patch_size,
    fastThreshold=fast_threshold
)



@torch.no_grad()
def get_latent_vectors(t_set, run_name):
    # t_set:pickle文件
    print('using {} point cloud'.format(PROCESS))
    print('using {} as features'.format(MODE))
    if HEIGHT:
        print('MAX: {}, MIN: {}'.format(TOP_Z_MAX,TOP_Z_MIN))
        assert TOP_Z_MIN < TOP_Z_MAX

    ks, ds = [], []
    Dict={}
    datafile = os.path.join(configs.data.evalset_folder, run_name+'_bev_'+'0.5_0.5_4_26_norm'+'.npy')
    import time
    print(time.time())
    bevs = np.load(datafile)[:,:,:,-1]#/np.log(16)*np.log(32) # 1->3 length,HW1

    for i, image in enumerate(bevs):
        image = (image * 255).astype(np.uint8)

        k ,d = extract_orb(orb, image) # N, 32
        ks.append(k)
        ds.append(d)
    print(time.time())

    return ks, ds


def evaluate_single_run():
    # Run evaluation on all eval datasets
    stats = pd.DataFrame(columns = ['F1max', 'R1', 'Sequence Length', 'Num. Revisits', 'Num. Correct Loc'])
    pickles = [pickle.load(open(os.path.join(configs.data.evalset_folder,i), 'rb')) for i in configs.eval.intra_files]
    import ipdb
    ipdb.set_trace()
    target_seq = {
        # 'V-03': pickles[0],
        # 'V-04': pickles[1],
        'K-03': pickles[2],
        # 'K-04' : pickles[3],
        # 'V-04': pickles_venman,
        # '3-02': pickles[0],
        # '5-03': pickles[0],
        # '5-06': pickles[1],
    }

    for name, database_set in target_seq.items():
        F1max, R1, seq_len, num_revisits, num_correct_loc = get_single_run_stats(database_set, name)
        stats.loc[name] = [F1max, R1, seq_len, num_revisits, num_correct_loc]
    return stats 

def euclidean_distance(query, database):
    return torch.cdist(torch.tensor(query).unsqueeze(0).unsqueeze(0), torch.tensor(database).unsqueeze(0)).squeeze().numpy()

def query_to_timestamp(query):
    base = os.path.basename(query)
    timestamp = float(base.replace('.pcd', ''))
    return timestamp

def get_single_run_stats(database_set, run_name, embeddings_k = [], embeddings_d = []):
    if len(embeddings_k) == 0:
        embeddings_k, embeddings_d = get_latent_vectors(database_set, run_name) # N x D, in chronological order
    timestamps = [query_to_timestamp(database_set[k]['query']) for k in range(len(database_set.keys()))]
    coords = np.array([[database_set[k]['easting'],database_set[k]['northing']] for k in range(len(database_set.keys()))])
    start_time = timestamps[0]

    world_thresh = configs.eval.world_thresh
    time_thresh = configs.eval.time_thresh

    # Thresholds, other trackers
    thresholds = np.linspace(1, 100, 100)
    num_thresholds = len(thresholds)

    num_true_positive = np.zeros(num_thresholds)
    num_false_positive = np.zeros(num_thresholds)
    num_true_negative = np.zeros(num_thresholds)
    num_false_negative = np.zeros(num_thresholds)

    num_revisits = 0
    num_correct_loc = 0

    for query_idx in tqdm(range(len(database_set)), desc = 'Evaluating Embeddings'):

        query_timestamp = timestamps[query_idx]
        query_coord = coords[query_idx]

        # Sanity check time 
        if (query_timestamp - start_time - time_thresh) < 0:
            continue 

        # Build retrieval database 
        tt = next(x[0] for x in enumerate(timestamps) if x[1] > (query_timestamp - time_thresh))
        seen_coords = coords[:tt+1] # Seen x 2

        # Get distances in feat space and real world
        dist_seen_world = euclidean_distance(query_coord, seen_coords)
        # Check if re-visit
        if np.any(dist_seen_world < world_thresh):
            revisit = True 
            num_revisits += 1
        else:
            revisit = False 
            continue

        num_neighbors = len(seen_coords)
        nn_ndx = np.array(range(0,num_neighbors))


        hbst = HBST(max_leaf_capacity=100, depth=256)
        for nn_i in range(num_neighbors):
            candidate_ndx = nn_i
            candidate_sc = embeddings_d[candidate_ndx]
            for dd in candidate_sc:
                hbst.insert(dd, nn_i)
        descriptors = embeddings_d[query_idx]
        vote = list(range(0,num_neighbors))
        for i, descriptor in enumerate(descriptors):
            closest_desc, cloest_idx, distance = hbst.find_closest(descriptor)
            if distance < 50:
                vote.append(cloest_idx)
            # vote and filter topN
        counter = Counter(vote)
        order = [num for num, count in sorted(counter.items(), key=lambda x: x[1], reverse=True)]
        nn_ndx = nn_ndx[order] # same to order

        # reranking
        num_neighbors = min(100,num_neighbors)
        sc_dist = np.zeros((num_neighbors,))
        for nn_i,candidate_ndx in enumerate(nn_ndx[:num_neighbors]):

            query_sc = embeddings_d[query_idx]
            candidate_sc = embeddings_d[candidate_ndx]
            matches = match_features(candidate_sc, query_sc)
            H,inliers = ransac_homography(embeddings_k[candidate_ndx],embeddings_k[query_idx],matches)
            if inliers >= 10:
                sc_dist[nn_i] = inliers

        reranking_order = np.argsort(-sc_dist)
        nn_ndx = nn_ndx[reranking_order]
        sc_dist = sc_dist[reranking_order]

        # Get top-1 candidate and distances in real world, embedding space 
        top1_idx = nn_ndx[0]
        top1_embed_dist = sc_dist[0]#dist_seen_embedding[top1_idx]
        top1_world_dist = dist_seen_world[top1_idx]

        if top1_world_dist < world_thresh:
            num_correct_loc += 1

        # Evaluate top-1 candidate 
        for thresh_idx in range(num_thresholds):
            threshold = thresholds[thresh_idx]

            if top1_embed_dist < threshold: # Positive Prediction
                if top1_world_dist < world_thresh:
                    num_true_positive[thresh_idx] += 1
                else: #elif top1_world_dist > 20:
                    num_false_positive[thresh_idx] += 1
            else: # Negative Prediction
                if not revisit:
                    num_true_negative[thresh_idx] += 1
                else:
                    num_false_negative[thresh_idx] += 1

    # Find F1Max and Recall@1 
    if num_revisits == 0:
        recall_1 = np.nan 
    else:
        recall_1 = num_correct_loc / num_revisits

    F1max = 0.0 
    for thresh_idx in range(num_thresholds):
        nTruePositive = num_true_positive[thresh_idx]
        nFalsePositive = num_false_positive[thresh_idx]
        nTrueNegative = num_true_negative[thresh_idx]
        nFalseNegative = num_false_negative[thresh_idx]

        nTotalTestPlaces = nTruePositive + nFalsePositive + nTrueNegative + nFalseNegative

        Precision = 0.0
        Recall = 0.0
        Prev_Recall = 0.0
        F1 = 0.0

        if nTruePositive > 0.0:
            Precision = nTruePositive / (nTruePositive + nFalsePositive)
            Recall = nTruePositive / (nTruePositive + nFalseNegative)
            F1 = 2 * Precision * Recall * (1/(Precision + Recall))

        if F1 > F1max:
            F1max = F1 
            thresh_max = thresholds[thresh_idx]

    print(f'Num Revisits : {num_revisits}')
    print(f'Num. Correct Locations : {num_correct_loc}')
    print(f'Sequence Length : {len(database_set)}')
    print(f'Recall@1: {recall_1}')
    print(f'F1max: {F1max}')


    return F1max, recall_1, len(database_set), num_revisits, num_correct_loc

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate ScanContext model')
    parser.add_argument('--config', type=str, required=False, default='sc_eval_config_wp.yaml')
    parser.add_argument('--save_dir', type = str, default = None)
    args, opts = parser.parse_known_args()

    configs.load(args.config, recursive = True)
    configs.update(opts)
    print('Training config path: {}'.format(args.config))

    # Evaluate 
    stats = evaluate_single_run()
    print(stats)
    if args.save_dir != None:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        stats.to_csv(os.path.join(args.save_dir, 'resutls_inrun.csv'), index = False)



