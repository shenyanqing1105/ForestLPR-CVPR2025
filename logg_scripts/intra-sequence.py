'''
for testing wild_places 
'''
import sys
sys.path.append("..")
import os 
import argparse 
import pandas as pd 
import numpy as np 
from tqdm import tqdm 
from utils import load_from_pickle, cosine_dist, euclidean_dist, query_to_timestamp
from logg3d_utils import get_latent_vectors
from models.pipeline_factory import get_pipeline
import torch
from scipy.spatial.distance import cdist
from os.path import join, exists, isfile
import matplotlib.pyplot as plt
from chamferdist import ChamferDistance
from logg3d_utils import EvalDataset

def eval_singlesession(database, embeddings, args):
    # Get embeddings, timestamps,coords and start time 
    database = load_from_pickle(database)
    # if embeddings != None:
    #     embeddings = load_from_pickle(embeddings)
    # else:
    model = get_pipeline(args.arch)
    save_path = args.resume_checkpoint
    checkpoint = torch.load(save_path)  # ,map_location='cuda:0')
    model.load_state_dict(checkpoint['model_state_dict'])

    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    model = model.cuda()
    model.eval()

    # embedding distance matrix
    Dataset = EvalDataset(database, args.root[0], arch=args.arch)
    embeddings = get_latent_vectors(model, database, args.root[0], 1024, arch=args.arch)
    matrix_cost = cdist(embeddings, embeddings)
    plt.figure(figsize=(20,20),dpi=150)
    plt.imshow(matrix_cost,cmap='gray_r',interpolation='nearest')
    plt.savefig(join('../plot', "matrix_intra_{}.png".format(args.run_names[0].split('/')[-1])))
    print('saving cost matrix')
    timestamps = [query_to_timestamp(database[k]['query']) for k in range(len(database.keys()))]
    # world distance matrix
    coords = np.array([[database[k]['easting'], database[k]['northing']] for k in range(len(database.keys()))])
    matrix_cost = cdist(coords, coords)
    matrix_cost = [[500 if element > args.world_thresh else element for element in row] for row in matrix_cost]
    plt.figure(figsize=(20,20),dpi=150)
    plt.imshow(matrix_cost,cmap='gray_r',interpolation='nearest')
    plt.savefig(join('../plot', "matrix_intra_GT_{}.png".format(args.run_names[0].split('/')[-1])))
    print('saving GT matrix')
    start_time = timestamps[0]

    # Thresholds, other trackers
    thresholds = np.linspace(0, 1, 1000)
    num_thresholds = len(thresholds)

    num_true_positive = np.zeros(num_thresholds)
    num_false_positive = np.zeros(num_thresholds)
    num_true_negative = np.zeros(num_thresholds)
    num_false_negative = np.zeros(num_thresholds)

    # Get similarity function 
    if args.similarity_function == 'cosine':
        dist_func = cosine_dist
    elif args.similarity_function == 'euclidean':
        dist_func = euclidean_dist
    else:
        raise ValueError(f'No supported distance function for {args.similarity_function}')

    num_revisits = 0
    num_falses = 0
    num_correct_loc = 0
    correct_values_x=[]
    correct_values_y=[]
    false_values_x=[]
    false_values_y=[]
    segments = []
    for query_idx in tqdm(range(len(database)), desc = 'Evaluating Embeddings'):
        q_embedding = embeddings[query_idx]
        q_timestamp = timestamps[query_idx]
        q_coord = coords[query_idx]

        # Exit if time elapsed since start is less than time threshold 
        
        if (q_timestamp - start_time - args.time_thresh) < 0:
            continue 

        # Build retrieval database 
        tt = next(x[0] for x in enumerate(timestamps) if x[1] > (q_timestamp - args.time_thresh))
        seen_embeddings = embeddings[:tt+1]
        seen_coords = coords[:tt+1]
        # Get distances in feature space and world 
        dist_seen_embedding = dist_func(q_embedding, seen_embeddings)
        dist_seen_world = euclidean_dist(q_coord, seen_coords)

        # Check if re-visit LCD
        if np.any(dist_seen_world < args.world_thresh):
            revisit = True
            gt_idx = np.argmin(dist_seen_world)
            num_revisits += 1
            # import ipdb
            # ipdb.set_trace()
        else:
            revisit = False

        # Get top-1 candidate and distances in real world, embedding space 一定找一个top1（只要满足时间要求）
        top1_idx = np.argmin(dist_seen_embedding)
        top1_embed_dist = dist_seen_embedding[top1_idx]
        top1_world_dist = dist_seen_world[top1_idx]

        if top1_world_dist < args.world_thresh:
            num_correct_loc += 1
            correct_values_x.append(q_coord[0])
            correct_values_y.append(q_coord[1])
            # print(top1_embed_dist)
        else:
            if revisit:
                import open3d as o3d
                qu = database[query_idx]['query']
                gt = database[gt_idx]['query']
                r1 = database[top1_idx]['query']
                # qpcd = o3d.io.read_point_cloud(os.path.join(args.root[0], qu))
                # xyz = np.asarray(qpcd.points).astype(np.float32)
                # q = torch.from_numpy(xyz)
                # qq = torch.unsqueeze(q, dim=0).cuda()
                # gpcd = o3d.io.read_point_cloud(os.path.join(args.root[0], gt))
                # xyz = np.asarray(gpcd.points).astype(np.float32)
                # g = torch.from_numpy(xyz)
                # gg = torch.unsqueeze(g, dim=0).cuda()
                # rpcd = o3d.io.read_point_cloud(os.path.join(args.root[0], r1))
                # xyz = np.asarray(rpcd.points).astype(np.float32)
                # r = torch.from_numpy(xyz)
                # rr = torch.unsqueeze(r, dim=0).cuda()
                # if chamferDist(qq,rr, bidirectional=True) < chamferDist(qq,gg, bidirectional=True):
                #     num_falses += 1
                #     print(top1_world_dist)
                # else:
                # import ipdb
                # ipdb.set_trace()

                false_values_x.append(q_coord[0])
                false_values_y.append(q_coord[1])
                segment = [tuple(q_coord), tuple(coords[top1_idx])]
                segments.append(segment)
        #         # print(top1_world_dist)
        #         # print(qu)
        #         # print(gt)
        #         # print(r1)
        #         # import ipdb
        #         # ipdb.set_trace()

        # Evaluate top-1 candidate 
        for thresh_idx in range(num_thresholds):
            threshold = thresholds[thresh_idx]

            if top1_embed_dist < threshold: # Positive Prediction
                if top1_world_dist < args.world_thresh:
                    num_true_positive[thresh_idx] += 1
                else:
                    num_false_positive[thresh_idx] += 1
            else: # Negative Prediction
                if not revisit:
                    num_true_negative[thresh_idx] += 1
                else:
                    num_false_negative[thresh_idx] += 1

    # Find Recall@1 
    recall_1 = num_correct_loc / num_revisits

    # plot the paths with loop, path+correct places+false places
    df = pd.read_csv(join(args.root[0],args.run_names[0],'poses_aligned.csv'), delimiter = ',', dtype = str)

    x_values = df['x'].values.tolist()
    x_values = [float(item) for item in x_values]
    y_values = df['y'].values.tolist()
    y_values = [float(item) for item in y_values]

    plt.figure(figsize=(20,20),dpi=150)
    plt.plot(x_values, y_values, marker='.', markersize=0.1, linestyle='-', color='b') # path
    plt.plot(correct_values_x, correct_values_y, linestyle='', marker='o', markersize=0.5, color='r')
    plt.plot(false_values_x, false_values_y, linestyle='', marker='o', markersize=0.5, color='y')
    plt.savefig(join('../plot', "path_{}.png".format(args.run_names[0].split('/')[-1])))

    # plot only for false pairs
    plt.figure(figsize=(30,30),dpi=150)
    import ipdb
    ipdb.set_trace()
    for segment in segments:
        x, y = zip(*segment)
        plt.plot(x,y,marker='.', markersize=0.2, linestyle='-', linewidth=0.2, color='b')
        # plt.scatter(*segment[0], c='yellow', s=0.5, zorder=3)
        # plt.scatter(*segment[1], c='green', s=0.5, zorder=3)
    plt.savefig(join('../plot', "pair_{}.png".format(args.run_names[0].split('/')[-1])))
    import ipdb
    ipdb.set_trace()
    # Find F1Max
    F1max = 0.0 
    for thresh_idx in range(num_thresholds):
        nTruePositive = num_true_positive[thresh_idx]
        nFalsePositive = num_false_positive[thresh_idx]
        nTrueNegative = num_true_negative[thresh_idx]
        nFalseNegative = num_false_negative[thresh_idx]

        Precision = 0.0
        Recall = 0.0
        F1 = 0.0

        if nTruePositive > 0.0:
            Precision = nTruePositive / (nTruePositive + nFalsePositive)
            Recall = nTruePositive / (nTruePositive + nFalseNegative)
            F1 = 2 * Precision * Recall * (1/(Precision + Recall))

        if F1 > F1max:
            F1max = F1 
            # print(nTruePositive, nFalsePositive, nTrueNegative, nFalseNegative)
    return {'F1max': F1max, 'Recall@1': recall_1, 'Sequence Length': len(embeddings), 'Num. Revisits': num_revisits, 'Num. Correct Locations': num_correct_loc}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Loading and saving paths 
    parser.add_argument('--root', default = '/opt/data/private/dataset/Wild-Places', type = str, nargs = '+', help = 'Path of database sets')
    parser.add_argument('--databases', required = True, type = str, nargs = '+', help = 'List of paths to pickles containing info about database sets')
    parser.add_argument('--database_features', default = None, type = str, nargs = '+', help = 'List of paths to pickles containing feature vectors for database sets')
    parser.add_argument('--run_names', type = str, nargs = '+', help = 'List of names of runs being evaluated')
    parser.add_argument('--save_dir', type = str, default = None, help = 'Save Directory for results csv')
    parser.add_argument('--resume_checkpoint', type = str, default = '../checkpoints/LoGG3D-Net.pth')
    parser.add_argument('--arch', type = str, default = 'LOGG3D')
    # Eval parameters
    parser.add_argument('--world_thresh', type = float, default = 3, help = 'Distance to be considered revisit in world')
    parser.add_argument('--time_thresh', type = float, default = 600, help = 'Time before a previous frame can be considered a valid revisit')
    parser.add_argument('--similarity_function', type = str, default = 'cosine', help = 'Distance function used to calculate similarity of embeddings')
    args = parser.parse_args()

    chamferDist = ChamferDistance()
    stats = pd.DataFrame(columns = ['F1max', 'Recall@1', 'Sequence Length', 'Num. Revisits', 'Num. Correct Locations'])
    for database, embeddings, location in zip(args.databases, args.database_features, args.run_names):
        temp_stats = eval_singlesession(database, embeddings, args)
        stats.loc[location] = [temp_stats['F1max'], temp_stats['Recall@1'], temp_stats['Sequence Length'], temp_stats['Num. Revisits'], temp_stats['Num. Correct Locations']]
    print(stats)

    if args.save_dir:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        stats.to_csv(os.path.join(args.save_dir, 'intra-run_results.csv'), index = False)


