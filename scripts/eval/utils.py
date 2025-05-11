import os 
import pickle 
import torch 
import numpy as np
import tqdm
from scripts.eval.logg3d.logg3d_utils import *
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.neighbors import KDTree


def query_to_timestamp(query):
    base = os.path.basename(query)
    timestamp = float(base.replace('.pcd', ''))
    return timestamp

def euclidean_dist(query, database):
    return torch.cdist(torch.tensor(query).unsqueeze(0).unsqueeze(0), torch.tensor(database).unsqueeze(0)).squeeze().numpy()

def cosine_dist(query, database):
    return np.array(1 - torch.einsum('D,ND->N', torch.tensor(query), torch.tensor(database)))

def load_from_pickle(pickle_path):
    with open(pickle_path, 'rb') as f:
        file = pickle.load(f)
    return file 

def intra_Recall_N(database, embeddings, args):
    # Get embeddings, timestamps,coords and start time 
    '''
    database: loaded pickle file
    '''

    timestamps = [query_to_timestamp(database[k]['query']) for k in range(len(database.keys()))]
    coords = np.array([[database[k]['easting'], database[k]['northing']] for k in range(len(database.keys()))])
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
    num_correct_loc = 0

    for query_idx in tqdm.tqdm(range(len(database)), desc = 'Evaluating Embeddings'):
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

        # Check if re-visit 
        if np.any(dist_seen_world < args.world_thresh):
            revisit = True 
            num_revisits += 1 
        else:
            revisit = False 

        # Get top-1 candidate and distances in real world, embedding space 
        top1_idx = np.argmin(dist_seen_embedding)
        top1_embed_dist = dist_seen_embedding[top1_idx]
        top1_world_dist = dist_seen_world[top1_idx]
        if revisit:
            if top1_world_dist < args.world_thresh:
                with open('output.txt', 'a') as file:
                    pcd = database[query_idx]['query'].split('/')[-1]
                    pcd1 = database[top1_idx]['query'].split('/')[-1]
                    file.write(f"{pcd},{pcd1},{top1_world_dist},{True},{query_idx},{top1_idx}\n")
                num_correct_loc += 1
            else:
                with open('output.txt', 'a') as file:
                    pcd = database[query_idx]['query'].split('/')[-1]
                    pcd1 = database[top1_idx]['query'].split('/')[-1]
                    file.write(f"{pcd},{pcd1},{top1_world_dist},{False},{query_idx},{top1_idx}\n")


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

    # Find F1Max and Recall@1 
    recall_1 = num_correct_loc / num_revisits

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

    return {'F1max': F1max, 'Recall@1': recall_1, 'Sequence Length': len(embeddings), 'Num. Revisits': num_revisits, 'Num. Correct Locations': num_correct_loc}

def inter_Recall_N(query_sets, database_sets, query_feat, database_feat, location, args):
    assert len(query_sets) == len(database_sets), f'Length of Query and Database Dictionaries is not the same: {len(query_sets)} vs {len(database_sets)}'

    stats = pd.DataFrame(columns = ['Recall@1', 'Recall@5', 'Recall@10', 'Recall@1%', 'MRR'])

    temp_stats = evaluate_location(query_sets, database_sets, query_feat, database_feat, location, args)
    stats.loc[location] = [temp_stats['ave_recall'][0], temp_stats['ave_recall'][4], temp_stats['ave_recall'][9], temp_stats['ave_one_percent_recall'], temp_stats['mrr']]

    stats.loc['Average'] = stats.mean(axis = 0)
    return stats


def evaluate_location(query_sets, database_sets, query_feat, database_feat, location, args):
    # Run evaluation on a single location 
    recall = np.zeros(25)
    count = 0 
    recall_1p = []
    mrr = []

    mrr_grid = np.identity(len(database_sets)) * 100
    recall_1_grid = np.identity(len(database_sets)) * 100
    pbar = tqdm.tqdm(total = len(query_sets)**2 - len(query_sets), desc = f'Eval: {location}')

    for i in range(len(query_sets)):
        for j in range(len(query_sets)):
            if i == j:
                continue 
            pbar.update(1)
            pair_recall, pair_recall_1p, pair_mrr = get_recall(i, j, database_feat, query_feat, database_sets, query_sets)

            recall += np.array(pair_recall)
            recall_1p.append(pair_recall_1p)
            mrr.append(pair_mrr)
            count += 1 

            recall_1_grid[i][j] = pair_recall[0]
            mrr_grid[i][j] = pair_mrr 

    ave_recall = recall / count
    ave_recall_1p = np.mean(recall_1p)
    ave_mrr = np.mean(mrr)

    
    stats = {'ave_recall': ave_recall, 'ave_one_percent_recall': ave_recall_1p, 'mrr': ave_mrr}
    print('printing status:', stats)
    import ipdb
    ipdb.set_trace()
    if args.plotPath != None:
        if not os.path.exists(args.plotPath):
            os.makedirs(args.plotPath)
        fig, (ax1, ax2) = plt.subplots(1,2)
        
        # Recall@1
        ax1 = sns.heatmap(recall_1_grid.T, linewidth = 2.5, cmap = 'YlGn', annot = np.round(recall_1_grid.T, decimals = 1),
                          fmt = '', annot_kws={"size": 20 }, cbar = False, square=True, ax = ax1)
        ax1.set_xlabel('Database')
        ax1.set_ylabel('Query')

        # MRR
        ax2 = sns.heatmap(mrr_grid.T, linewidth = 2.5, cmap = 'YlGn', annot = np.round(mrr_grid.T, decimals = 1),
                          fmt = '', annot_kws={"size": 20 }, cbar = False, square=True, ax = ax2)
        ax2.set_xlabel('Database')
        ax2.set_ylabel('Query')

        plt.savefig(os.path.join(args.plotPath, f"results_grid_{location}.png"), pad_inches = 0.1, dpi = 300)

    return stats



def get_recall(m, n, database_feat, query_feat, database_sets, query_sets):

    database_feat_run = database_feat[m]
    queries_feat_run = query_feat[n]

    # Get database info 
    database_nbrs = KDTree(database_feat_run)

    # Set up variables 
    num_neighbours = 25 
    recall = np.zeros(num_neighbours)
    recall_idx = []

    one_percent_retrieved = 0 
    threshold = max(int(round(len(database_feat_run) / 100)), 1)

    num_evaluated = 0 

    for i in range(len(queries_feat_run)):
        query_details = query_sets[n][i]
        true_neighbours = query_details[m]
        if len(true_neighbours) == 0:
            continue 
        num_evaluated += 1 

        # Find nearest neighbours 
        _, indices = database_nbrs.query(np.asarray([queries_feat_run[i]]), k = num_neighbours)

        for j in range(len(indices[0])):
            if indices[0][j] in true_neighbours:
                recall[j] += 1 
                recall_idx.append(j + 1)
                break 

        if len(list(set(indices[0][0:threshold]).intersection(set(true_neighbours)))) > 0:
            one_percent_retrieved += 1

    recall = np.cumsum(recall) / float(num_evaluated) * 100
    recall_1p = one_percent_retrieved / float(num_evaluated) * 100 
    mrr = np.mean(1 / np.array(recall_idx)) * 100 

    return recall, recall_1p, mrr 
