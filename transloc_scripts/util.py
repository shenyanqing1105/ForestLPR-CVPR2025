# coding=utf-8
'''
Author: shenyanqing1105 1159364090@qq.com
Date: 2024-04-25 17:01:01
LastEditors: shenyanqing1105 1159364090@qq.com
LastEditTime: 2024-07-16 20:07:39
FilePath: /TransLoc3D/scripts/util.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
# coding=utf-8
'''
Author: shenyanqing
Description: 
'''
import sys
sys.path.append("..")
import os 
import pickle 
import torch 
import numpy as np 
from transloc3d_utils import *


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

def get_latent_vectors(database):
    # Placeholder function for user 
    # raise NotImplementedError("No method for feature extraction currently implemented")
    pass

class TrainingTuple:
    # Tuple describing an element for training/validation
    def __init__(self, id: int, timestamp: int, rel_scan_filepath: str, positives: np.ndarray,
                 non_negatives: np.ndarray, position: np.ndarray):
        # id: element id (ids start from 0 and are consecutive numbers)
        # ts: timestamp
        # rel_scan_filepath: relative path to the scan
        # positives: sorted ndarray of positive elements id
        # negatives: sorted ndarray of elements id
        # position: x, y position in meters (northing, easting)
        assert position.shape == (2,)

        self.id = id
        self.timestamp = timestamp
        self.rel_scan_filepath = rel_scan_filepath
        self.positives = positives
        self.non_negatives = non_negatives
        self.position = position

def Recall_at_N(predictions, gt):
    '''
    predictions : rank index matrix for queries according to the similarity
    gt : list of list
    write_TBoard : validate(True) test(False)
    '''
    n_values = [1, 5, 10, 15, 20, 100]
    # if NMS:
    #     predictions_new = []
    #     for qIx, pred in enumerate(predictions):
    #         pred = np.array(pred)
    #         _, idx = np.unique(np.floor(pred / 12).astype(np.int), return_index=True)
    #         pred = pred[np.sort(idx)]
    #         pred = pred[:max(n_values)]
    #         predictions_new.append(pred)
    #     predictions = np.array(predictions_new)
    correct_at_n = np.zeros(len(n_values))
    for qIx, pred in enumerate(predictions):  # every row
        for i, n in enumerate(n_values):
            if np.any(np.in1d(pred[:n], gt[qIx])):  # 存在即可,从query角度
                correct_at_n[i:] += 1
                break
        if qIx % 1000 == 0:
            print("==> Number ({}/{})".format(qIx,len(predictions)), flush=True)

    numQ = predictions.shape[0]
    recall_at_n = correct_at_n / numQ

    recalls = {}
    for i, n in enumerate(n_values):
        recalls[n] = recall_at_n[i]
        print("====> Recall@{}: {:.4f}".format(n, recall_at_n[i]))

    return recalls