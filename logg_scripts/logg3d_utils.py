import os
import numpy as np
import tqdm
from torchsparse import SparseTensor
from torch.utils.data import DataLoader
import open3d as o3d
from torchsparse.utils.quantize import sparse_quantize
from torchsparse.utils.collate import sparse_collate
import math


def clip(lidar):
    TOP_Z_MIN = 1.0
    TOP_Z_MAX = 6.0

    idx = np.where (lidar[:,2]>TOP_Z_MIN)
    lidar = lidar[idx]
    idx = np.where (lidar[:,2]<TOP_Z_MAX)
    lidar = lidar[idx]
    return lidar

def make_sparse_tensor(lidar_pc, voxel_size=0.05, return_points=False):
    # get rounded coordinates
    coords = np.round(lidar_pc[:, :3] / voxel_size) # 量化
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

    if isinstance(list_data, SparseTensor):
        return list_data
    else:
        return sparse_collate(list_data)

class EvalDataset:
    def __init__(self, dataset, dataset_folder, arch='LOGG3D'):
        self.set = dataset
        self.dataset_folder = dataset_folder
        self.arch = arch
    def __len__(self):
        return len(self.set)

    def get_pointcloud_tensor(self, fname):
        pcd = o3d.io.read_point_cloud(fname)
        fname = fname.replace('downsampled', 'normalized')
        # downpcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)
        xyz = np.asarray(pcd.points)
        # xyz = clip(xyz)
        oo = np.ones(len(xyz)).reshape((-1,1))
        xyzr = np.hstack((xyz, oo)).astype(np.float32)
        if self.arch == 'LOGG3D':
            return make_sparse_tensor(xyzr, 0.5)
        else:
            return xyzr

    def __getitem__(self, idx):
        path = os.path.join(self.dataset_folder, self.set[idx]['query'])
        xyz0_th = self.get_pointcloud_tensor(path)

        return xyz0_th

def get_eval_dataloader(dataset, dataset_folder,arch):
    dataset = EvalDataset(dataset, dataset_folder,arch)
    dataloader = DataLoader(
        dataset,
        batch_size = 1,
        shuffle = False,
        collate_fn = sparcify_and_collate_list,#collate_sparse_tuple,
        num_workers = 4
    )
    return dataloader

def get_latent_vectors(model, dataset, dataset_folder, gd_dim=512, arch='logg'):
    eval_dataloader = get_eval_dataloader(dataset, dataset_folder, arch)
    vectors = []
    model.eval()

    for idx, batch in tqdm.tqdm(enumerate(eval_dataloader), desc = 'Getting Latent Vectors', total = math.ceil(len(eval_dataloader.dataset))):
        # batch = {k:v.to('cuda') for k,v in batch.items()}
        batch = batch.to('cuda')
        y = model(batch)
        gds = y[0].detach().cpu().numpy()
        gds = gds.reshape((-1,gd_dim))
        vectors.append(gds)

    vectors = np.concatenate(vectors, 0)
    return vectors