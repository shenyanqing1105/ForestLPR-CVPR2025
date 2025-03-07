# coding=utf-8
'''
Author: shenyanqing
Description: 
'''
import sys
sys.path.append("..")
import os
import numpy as np
import tqdm
from torchsparse import SparseTensor
from torch.utils.data import DataLoader
import open3d as o3d
from torchsparse.utils.quantize import sparse_quantize
from torchsparse.utils.collate import sparse_collate
import math
from datasets.utils.collate_fns import create_collate_fn


def make_sparse_tensor(lidar_pc, voxel_size=0.05, return_points=False):
    # get rounded coordinates
    coords = np.round(lidar_pc[:, :3] / voxel_size)
    coords -= coords.min(0, keepdims=1)
    feats = lidar_pc

    # sparse quantization: filter out duplicate points
    _, indices = sparse_quantize(coords, return_index=True)
    coords = coords[indices]
    feats = feats[indices]

    # construct the sparse tensor
    inputs = SparseTensor(feats, coords)
    inputs = sparse_collate([inputs])
    inputs.C = inputs.C.int()
    if return_points:
        return inputs, feats
    else:
        return inputs

def sparcify_and_collate_list(list_data):
    # print(list_data)

    if isinstance(list_data, SparseTensor):
        return list_data
    else:
        print(sparse_collate(list_data))
        return sparse_collate(list_data)

class EvalDataset:
    def __init__(self, dataset, dataset_folder):
        self.set = dataset
        self.dataset_folder = dataset_folder

    def __len__(self):
        return len(self.set)

    def get_pointcloud_tensor(self, fname):
        pcd = o3d.io.read_point_cloud(fname)
        # downpcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)
        xyz = np.asarray(pcd.points)
        oo = np.ones(len(xyz)).reshape((-1,1))
        xyzr = np.hstack((xyz, oo)).astype(np.float32)

        return make_sparse_tensor(xyzr, 0.5)

    def __getitem__(self, idx):
        path =  os.path.join(self.dataset_folder, self.set[idx]['query'])
        xyz0_th = self.get_pointcloud_tensor(path)

        return xyz0_th

def get_eval_dataloader(dataset, dataset_folder, cfg):
    dataset = EvalDataset(dataset, dataset_folder)
    dataloader = DataLoader(
        dataset,
        batch_size = 4,
        shuffle = False,
        collate_fn = create_collate_fn(
                dataset, cfg.model_cfg.quantization_size if hasattr(cfg.model_cfg, 'quantization_size') else None, True),#collate_sparse_tuple,
        num_workers = 4
    )
    return dataloader

def get_latent_vectors(model, dataset, dataset_folder, cfg):
    eval_dataloader = get_eval_dataloader(dataset, dataset_folder, cfg)
    vectors = []
    model.eval()
    
    for idx, batch in tqdm.tqdm(enumerate(eval_dataloader), desc = 'Getting Latent Vectors', total = math.ceil(len(eval_dataloader.dataset) / 16)):
        batch = batch.to('cuda')
        y = model(batch)
        gds = y[0].detach().cpu().numpy()
        vectors.append(gds)

    vectors = np.concatenate(vectors, 0)
    return vectors 

