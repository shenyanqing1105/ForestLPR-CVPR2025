# coding=utf-8
'''
Author: shenyanqing
Description: 
'''
import os
import numpy as np
import tqdm
import torch
import MinkowskiEngine as ME
import math
import open3d as o3d

def get_latent_vectors(model, dataset, dataset_folder, device, params):
    vectors = []
    model.eval()

    for idx in tqdm.tqdm(dataset, desc = 'Getting Latent Vectors', total = math.ceil(len(dataset))):
        if isinstance(dataset[idx],dict):
            path =  os.path.join(dataset_folder, dataset[idx]['query'])
        else:
            path =  os.path.join(dataset_folder, dataset[idx].rel_scan_filepath)
        # path =  os.path.join(dataset_folder, dataset[idx]['query'])
        pcd = o3d.io.read_point_cloud(path)
        xyz = np.asarray(pcd.points).astype(np.float32)

        # mask = np.all(np.isclose(xyz, 0), axis=1)
        # xyz = xyz[~mask]
        # mask = xyz[:, 2] > 0.1 #滤地面
        # xyz = xyz[mask]
        pc = torch.tensor(xyz)
        pc = pc.to('cuda')
        embedding = compute_embedding(model, pc, device, params)
        vectors.append(embedding)

    vectors = np.concatenate(vectors, 0)
    return vectors

def compute_embedding(model, pc, device, params):
    coords, _ = params.model_params.quantizer(pc) # 乘100，并量化
    with torch.no_grad():
        bcoords = ME.utils.batched_coordinates([coords])
        feats = torch.ones((bcoords.shape[0], 1), dtype=torch.float32)
        batch = {'coords': bcoords.to(device), 'features': feats.to(device)}
        # Compute global descriptor
        y = model(batch)
        embedding = y['global'].detach().cpu().numpy()

    return embedding

def getPositives(set): # list of list
    import ipdb
    ipdb.set_trace()
    positives_=[]
    near_positives=[]
    for index in range(len(set)):
        pos = []
        near_pos=[index]
        positives = set[index].positives

        for i in positives:
        # for i in self.set[index].positives.tolist():
            if abs(set[i].timestamp - set[index].timestamp) > 20: # 避免nearby 增加多样性
                pos.append(i)
            else:
                near_pos.append(i)
        positives_.append(pos)
        near_positives.append(near_pos)
    return positives_, near_positives