# coding=utf-8
'''
Author: shenyanqing1105 1159364090@qq.com
Date: 2025-03-05 16:28:34
LastEditors: shenyanqing1105 1159364090@qq.com
LastEditTime: 2025-03-05 16:38:14
FilePath: /ForestLPR-CVPR2025/transloc_scripts/transloc3d_utils.py
'''
import sys
sys.path.append("..")
import os
import numpy as np
import tqdm
import torch
import open3d as o3d
import math
import torch.nn.functional as F
import MinkowskiEngine as ME
import matplotlib.pyplot as plt

def get_latent_vectors(model, dataset, dataset_folder, cfg):
    vectors = []
    bevs = []
    model.eval()
    for idx in tqdm.tqdm(dataset, desc = 'Getting Latent Vectors', total = math.ceil(len(dataset))):
        if isinstance(dataset[idx],dict):
            path =  os.path.join(dataset_folder, dataset[idx]['query'])
        else:
            path =  os.path.join(dataset_folder, dataset[idx].rel_scan_filepath)
        path = path.replace('downsampled', 'normalized')
        pcd = o3d.io.read_point_cloud(path)
        xyz = np.asarray(pcd.points).astype(np.float32)
        tops=[]
        pc = torch.tensor(xyz)
        pc = pc.to('cuda')
        embedding = compute_embedding(model, pc, idx, cfg)
        vectors.append(embedding)
        bevs.append(tops)
    vectors = np.concatenate(vectors, 0)
    # bevs = np.stack(bevs)
    # np.save('../data/bev_1_1_0.5.npy', bevs)

    return vectors

def compute_embedding(model, pc, idx, cfg):
    with torch.no_grad():
        # Compute global descriptor
        # meta = {'idx': idx, 'filename': path}
        data = {'pcd': pc}
        coords = ME.utils.sparse_quantize(coordinates=pc, quantization_size=cfg.model_cfg.quantization_size)
        coords = ME.utils.batched_coordinates([coords])
        feats = torch.ones((coords.shape[0], 1), dtype=torch.float32)
        batch = {'coords': coords.to('cuda'), 'features': feats.to('cuda')}

        y = model(batch)
        y = F.normalize(y, p=2, dim=1)
        embedding = y.detach().cpu().numpy()

    return embedding

def getPositives(set): # list of list
    positives_=[]
    near_positives=[]
    for index in range(len(set)):
        pos = []
        near_pos=[index]
        positives = set[index].positives

        for i in positives:
        # for i in self.set[index].positives.tolist():
            if abs(set[i].timestamp - set[index].timestamp) > 20:
                pos.append(i)
            else:
                near_pos.append(i)
        positives_.append(pos)
        near_positives.append(near_pos)
    return positives_, near_positives