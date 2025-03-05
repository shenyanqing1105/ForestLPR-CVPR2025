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
from minkloc_utils import get_latent_vectors, getPositives
from scripts.util import load_from_pickle, cosine_dist, euclidean_dist, query_to_timestamp, Recall_at_N
from models.model_factory import model_factory
import torch
from misc.utils import TrainingParams
import faiss

def eval_singlesession(database, embeddings, args):
    # Get embeddings, timestamps,coords and start time 

    database = load_from_pickle(database)

    params = TrainingParams(args.config, args.model_config, debug=args.debug)
    params.print()
    model = model_factory(params.model_params)
    if args.weights is not None:
        assert os.path.exists(args.weights), 'Cannot open network weights: {}'.format(args.weights)
        print('Loading weights: {}'.format(args.weights))
    checkpoint = torch.load(args.weights)  # ,map_location='cuda:0')
    model.load_state_dict(checkpoint)
    if not torch.cuda.is_available():
        print("CHECK CUDA")
        return
    model = model.cuda()
    model.eval()
    device ='cuda'
    # embedding distance matrix
    embeddings = get_latent_vectors(model, database, args.root[0], device, params)


    if args.val_mode:
        qFeat = embeddings.astype('float32')
        dbFeat = embeddings.astype('float32')

        n_values = [1, 5, 10, 20, 100]
        print('====> Building faiss index')
        faiss_index = faiss.IndexFlatL2(params.model_params.feature_size)
        faiss_index.add(dbFeat)

        print('====> Calculating recall @ N [1,5,10,20,100]')

        _, predictions = faiss_index.search(
            qFeat, max(n_values))
        gt, values = getPositives(database)

        for qIx, pred in enumerate(predictions):
            indexes = np.where(np.isin(pred, values[qIx]))[0]
            rest_elements = np.delete(pred,indexes)
            moved_elements = np.intersect1d(pred,values[qIx])
            if len(rest_elements)>0:
                predictions[qIx,:] = np.append(rest_elements, moved_elements)
            else:
                gt[qIx] = predictions[qIx].tolist()
        metrics = Recall_at_N(predictions,gt)
        return metrics
    timestamps = [query_to_timestamp(database[k]['query']) for k in range(len(database.keys()))]
    # # world distance matrix
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
        else:
            revisit = False 

        # Get top-1 candidate and distances in real world, embedding space
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
                qu = database[query_idx]['query']
                gt = database[gt_idx]['query']
                r1 = database[top1_idx]['query']
                false_values_x.append(q_coord[0])
                false_values_y.append(q_coord[1])
                segment = [tuple(q_coord), tuple(coords[top1_idx])]
                segments.append(segment)


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

    return {'F1max': F1max, 'Recall@1': recall_1, 'Sequence Length': len(embeddings), 'Num. Revisits': num_revisits, 'Num. Correct Locations': num_correct_loc}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Loading and saving paths 
    parser.add_argument('--root', default = '/opt/data/private/dataset/Wild-Places', type = str, nargs = '+', help = 'Path of database sets')
    parser.add_argument('--databases', required = True, type = str, nargs = '+', help = 'List of paths to pickles containing info about database sets')
    parser.add_argument('--database_features', default = None, type = str, nargs = '+', help = 'List of paths to pickles containing feature vectors for database sets')
    parser.add_argument('--run_names', type = str, nargs = '+', help = 'List of names of runs being evaluated')
    parser.add_argument('--save_dir', type = str, default = None, help = 'Save Directory for results csv')
    # Eval parameters
    parser.add_argument('--world_thresh', type = float, default = 3, help = 'Distance to be considered revisit in world')
    parser.add_argument('--time_thresh', type = float, default = 600, help = 'Time before a previous frame can be considered a valid revisit')
    parser.add_argument('--similarity_function', type = str, default = 'cosine', help = 'Distance function used to calculate similarity of embeddings')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--model_config', type=str, required=True, help='Path to the model-specific configuration file')
    parser.add_argument('--weights', type=str, required=False, help='Trained model weights')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--val_mode', action='store_true')
    parser.set_defaults(debug=False)
    parser.add_argument('--visualize', dest='visualize', action='store_true')
    parser.set_defaults(visualize=False)
    parser.add_argument('--log', dest='log', action='store_true')
    parser.set_defaults(log=False)
    args = parser.parse_args()

    stats = pd.DataFrame(columns = ['F1max', 'Recall@1', 'Sequence Length', 'Num. Revisits', 'Num. Correct Locations'])
    for database, embeddings, location in zip(args.databases, args.database_features, args.run_names):
        temp_stats = eval_singlesession(database, embeddings, args)
        stats.loc[location] = [temp_stats['F1max'], temp_stats['Recall@1'], temp_stats['Sequence Length'], temp_stats['Num. Revisits'], temp_stats['Num. Correct Locations']]
    print(stats)

    if args.save_dir:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        stats.to_csv(os.path.join(args.save_dir, 'intra-run_results.csv'), index = False)


