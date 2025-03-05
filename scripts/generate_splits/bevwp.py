'''
original version
'''
import os
import numpy as np
import tqdm
import torch
import open3d as o3d
import math
import pickle
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.utils.data as data
import h5py
from sklearn.neighbors import NearestNeighbors


root_dir = '/../anymal'
pf = '/../Wild-Places/training/training_wild-places.pickle'
pf_test = '/../Wild-Places/training/testing_wild-places.pickle'

USE_NORM = True
TOP_X_DIVISION = 0.5
TOP_Y_DIVISION = 0.5
TOP_Z_DIVISION = 5
HEIGHT_Z =16
TOP_Z_MIN = 1.0
TOP_Z_MAX =6.0
MODE_ = 'density'
RESOLUTION='{}_{}_{}_{}_norm'.format(TOP_X_DIVISION,TOP_Y_DIVISION,TOP_Z_DIVISION,HEIGHT_Z)

def lidar_to_top(lidar):
    '''Generates the BEV maps dictionary. One density/height map is created for
        the whole point cloud or each slice of the point cloud.'''

    TOP_Y_MIN = -30
    TOP_Y_MAX = +30
    TOP_X_MIN = -30
    TOP_X_MAX = 30


    idx = np.where (lidar[:,0]>TOP_X_MIN)
    lidar = lidar[idx]
    idx = np.where (lidar[:,0]<TOP_X_MAX)
    lidar = lidar[idx]

    idx = np.where (lidar[:,1]>TOP_Y_MIN)
    lidar = lidar[idx]
    idx = np.where (lidar[:,1]<TOP_Y_MAX)
    lidar = lidar[idx]

    idx = np.where (lidar[:,2]>TOP_Z_MIN)
    lidar = lidar[idx]
    idx = np.where (lidar[:,2]<TOP_Z_MAX)
    lidar = lidar[idx]

    pxs=lidar[:,0]
    pys=lidar[:,1]
    pzs=lidar[:,2]
    qxs=((pxs-TOP_X_MIN)//TOP_X_DIVISION).astype(np.int32)
    qys=((pys-TOP_Y_MIN)//TOP_Y_DIVISION).astype(np.int32)
    #qzs=((pzs-TOP_Z_MIN)//TOP_Z_DIVISION).astype(np.int32)
    qzs=(pzs-TOP_Z_MIN)/TOP_Z_DIVISION
    quantized = np.dstack((qxs,qys,qzs)).squeeze()

    X0, Xn = 0, int((TOP_X_MAX-TOP_X_MIN)//TOP_X_DIVISION)+1
    Y0, Yn = 0, int((TOP_Y_MAX-TOP_Y_MIN)//TOP_Y_DIVISION)+1
    Z0, Zn = 0, int((TOP_Z_MAX-TOP_Z_MIN)/TOP_Z_DIVISION) # slices
    height  = Xn - X0
    width   = Yn - Y0
    channel = Zn - Z0  + 1
    top = np.zeros(shape=(height,width,channel), dtype=np.float32)

    for x in range(Xn):
        ix  = np.where(quantized[:,0]==x)
        quantized_x = quantized[ix]
        if len(quantized_x) == 0 : continue
        yy = -x

        for y in range(Yn):
            iy  = np.where(quantized_x[:,1]==y)
            quantized_xy = quantized_x[iy]
            count = len(quantized_xy)
            if  count==0 : continue
            xx = -y

            # single BEV info
            if MODE_ == 'density':
                top[yy,xx,Zn] = np.log(count+1)
            elif MODE_ == 'height':
                top[yy,xx,Zn] = max(0,np.max(quantized_xy[:,2]))

            # multiple BEV info
            for z in range(Zn):
                iz = np.where ((quantized_xy[:,2]>=z) & (quantized_xy[:,2]<=z+1))
                quantized_xyz = quantized_xy[iz]
                if len(quantized_xyz) == 0 : continue
                zz = z
                if MODE_ == 'density':
                    top[yy,xx,zz] = np.log(len(quantized_xyz)+1)
                elif MODE_ == 'height':
                    top[yy,xx,zz] = max(0,np.max(quantized_xyz[:,2]))
    return top,channel

def transform(img_size):
    return transforms.Compose([
        transforms.Resize(img_size, antialias=True)
    ])
def input_transform(img_size=(480,480)):
    return transforms.Compose([
        transforms.Resize(img_size, antialias=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
    ])

def generate_bevs(set,dataset_folder, file):
    bevs = []
    for idx in tqdm.tqdm(set, desc = 'Generating bev tensors', total = math.ceil(len(set))):
        if isinstance(set[idx],dict):
            path =  os.path.join(dataset_folder, set[idx]['query'])
        else:
            path =  os.path.join(dataset_folder, set[idx].rel_scan_filepath)
        if 'norm' in file:
            path = path.replace('Clouds_downsampled','Clouds_normalized')
        pcd = o3d.io.read_point_cloud(path)
        xyz = np.asarray(pcd.points).astype(np.float32)
        tops=[]
        tops, n_height_maps = lidar_to_top(xyz)

        bevs.append(tops)
    bevs = np.stack(bevs)
    np.save(file, bevs)

def collate_fn(batch):
    """Creates mini-batch tensors from the list of tuples (query, positive, negatives).

    Args:
        data: list of tuple (query, positive, negatives). 
            - query: torch tensor of shape (3, h, w).
            - positive: torch tensor of shape (3, h, w).
            - negative: torch tensor of shape (n, 3, h, w).
    Returns:
        query: torch tensor of shape (batch_size, 3, h, w).
        positive: torch tensor of shape (batch_size, 3, h, w).
        negatives: torch tensor of shape (batch_size, n, 3, h, w).
    """
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        print('none')
        return None, None, None, None, None

    query, positive, negatives, indices = zip(
        *batch)
    query = data.dataloader.default_collate(query)
    positive = data.dataloader.default_collate(positive)
    negCounts = data.dataloader.default_collate(
        [x.shape[0] for x in negatives])
    negatives = torch.cat(negatives, 0)
    import itertools
    indices = list(itertools.chain(*indices))
    return query, positive, negatives, negCounts, indices

class TrainDataset(Dataset):
    def __init__(self, img_size, inputc, nNegSample=1000, nNeg=5, margin=0.1, dataset=pf, dataset_folder=root_dir):
        self.dataset = 'wp'
        with open(dataset, 'rb') as f:
            self.set = pickle.load(f)
        self.dataset_folder = dataset_folder
        self.datafile = os.path.join(dataset_folder, 'training/training_bev_'+RESOLUTION+'.npy')
        if MODE_ == 'height':
            self.datafile = os.path.join(dataset_folder, 'training/training_height_'+RESOLUTION+'.npy')
        if not os.path.exists(self.datafile):
            generate_bevs(self.set,self.dataset_folder, self.datafile)
        print("loading training bev file {}".format(self.datafile))
        if inputc == 1:
            self.bevs = np.load(self.datafile)[:,:,:,-1:]
            if USE_NORM:
                L, H, W, _ = self.bevs.shape # LHW1
                for i in range(L):
                    sub_tensor = self.bevs[i, :]
                    self.bevs[i, :] = sub_tensor / np.max(sub_tensor) # max-norm

        else:
            self.bevs = np.load(self.datafile)
            if USE_NORM:
                L, H, W, S = self.bevs.shape
                for i in range(L):
                    sub_tensor = self.bevs[i, :]
                    self.bevs[i, :] = sub_tensor / np.max(sub_tensor) # max-norm

        assert len(self.set) == len(self.bevs)

        if img_size:
            self.input_transform = transform(img_size)

        self.Featcache = None
        self.nNegSample = nNegSample
        self.nNeg = nNeg
        self.margin = margin
        self.start_idx = None
        self.negSample = None
        self.negNN = None
        self.violatingNeg = None
        self.negCache = [np.empty((0,), dtype=np.int64) for _ in range(
            len(self.set))]
        self.positives = [self.set[index].positives for index in range(len(self.set))]
        self.positives_conf = [self.set[index].positives_conf for index in range(len(self.set))]
        for i, pos in enumerate(self.positives):
            positives=pos
            self.positives[i] = [p for p in positives if abs(p-i) > 100]
            if not self.positives[i]:
                self.positives[i] = positives
        potential_positives = [self.set[index].non_negatives for index in range(len(self.set))]
        self.potential_negatives = []
        for pos in potential_positives:
            self.potential_negatives.append(np.setdiff1d(np.arange(len(self.set)),
                                                        pos, assume_unique=True))
    def __len__(self):
        return len(self.set) # pickle file

    def __getitem__(self, index):
        with h5py.File(self.Featcache, mode='r') as h5:
            h5feat = h5.get("features")
            qFeat = h5feat[index]
            posFeat = h5feat[self.positives[index]]
            knn = NearestNeighbors(n_jobs=1)
            knn.fit(posFeat)
            dPos, posNN = knn.kneighbors(qFeat.reshape(1, -1), 1)
            dPos = dPos.reshape(-1)
            posNN = posNN.reshape(-1)

            dPos = dPos[0].item()
            posIndex = self.positives[index][posNN[0]].item()

            negSample = np.random.choice(
                self.potential_negatives[index], self.nNegSample)
            negSample = np.unique(np.concatenate(
                [self.negCache[index], negSample]))

            negFeat = h5feat[negSample.tolist()]
            knn.fit(negFeat)

            dNeg, negNN = knn.kneighbors(qFeat.reshape(1, -1),
                                         self.nNeg*10)
            dNeg = dNeg.reshape(-1)
            negNN = negNN.reshape(-1)
            violatingNeg = dNeg < dPos + self.margin
            if np.sum(violatingNeg) < 1:
                return None

            negNN = negNN[violatingNeg][:self.nNeg]
            negIndices = negSample[negNN].astype(np.int32)
            self.negCache[index] = negIndices


        # query positive
        query = self.bevs[index]
        positive = self.bevs[posIndex]
        query = torch.tensor(query).permute(2,0,1)
        query = query.unsqueeze(0)
        positive = torch.tensor(positive).permute(2,0,1)
        positive = positive.unsqueeze(0)

        if self.input_transform:
            query = self.input_transform(query)
            positive = self.input_transform(positive)
            query = query.squeeze(0)
            positive = positive.squeeze(0)

        # negative
        negatives = []
        for negIndex in negIndices:
            negative = self.bevs[negIndex]
            negative = torch.tensor(negative).permute(2,0,1)
            negative = negative.unsqueeze(0)
            if self.input_transform:
                negative = self.input_transform(negative)
                negative = negative.squeeze(0)
            negatives.append(negative)
        negatives = torch.stack(negatives, 0)

        return query, positive, negatives, negIndices.tolist()


class DevSet(Dataset):
    def __init__(self, inputc, img_size, subset):
        self.dataset_folder = root_dir
        self.positives = None
        self.near_positives = None
        if img_size:
            self.input_transform = transform(img_size)

        if subset == 'val' or subset == 'train':
            with open(pf, 'rb') as f:
                self.set = pickle.load(f)
            self.subset = 'train_val'
            self.datafile = os.path.join(self.dataset_folder, 'training/training_bev_'+RESOLUTION+'.npy')
            if MODE_ == 'height':
                self.datafile = os.path.join(self.dataset_folder, 'training/training_height_'+RESOLUTION+'.npy')
        elif subset == 'test':
            with open(pf_test, 'rb') as f:
                self.set = pickle.load(f)
            self.datafile = os.path.join(self.dataset_folder, 'training/testing_bev_'+RESOLUTION+'.npy') # todo
            if MODE_ == 'height':
                self.datafile = os.path.join(self.dataset_folder, 'training/testing_height_'+RESOLUTION+'.npy')
            self.subset = 'test'
        print("loading bev file {}".format(self.datafile))

        self.dataset='wp'
        if not os.path.exists(self.datafile):
            generate_bevs(self.set,self.dataset_folder, self.datafile)
        if inputc == 1:
            self.bevs = np.load(self.datafile)[:,:,:,-1:]
            if USE_NORM:
                L, H, W, _ = self.bevs.shape
                for i in range(L):
                    sub_tensor = self.bevs[i, :]
                    self.bevs[i, :] = sub_tensor / np.max(sub_tensor)
        else:
            self.bevs = np.load(self.datafile)

    def __len__(self):
        return len(self.set)

    def __getitem__(self, index):
        img = self.bevs[index]
        img = torch.tensor(img).permute(2,0,1)
        img = img.unsqueeze(0)
        if self.input_transform:
            img = self.input_transform(img)
            img = img.squeeze(0)
        return img, index
    def getPositives(self): # list of list
        if self.positives is None:
            self.positives=[]
            self.near_positives=[]
            for index in range(len(self.set)):
                pos = []
                near_pos=[index]
                positives = self.set[index].positives
                for i in positives:
                    if abs(self.set[i].timestamp - self.set[index].timestamp) > 20:
                        pos.append(i)
                    else:
                        near_pos.append(i)
                self.positives.append(pos)
                self.near_positives.append(near_pos)
        return self.positives, self.near_positives

class TestSet(DevSet):
    def __init__(self, inputc, subset='test', img_size=None):
        super(TestSet, self).__init__(inputc, img_size, subset)


class ValSet(DevSet):
    def __init__(self, inputc, subset='val', img_size=None):
        super(ValSet, self).__init__(inputc, img_size, subset)

class IntraSet(Dataset):
    def __init__(self, inputc, img_size, subset):
        self.dataset_folder = root_dir
        self.positives = None
        if img_size:
            self.input_transform = transform(img_size)
        else:
            self.input_transform = None

        with open(os.path.join(root_dir,'testing',subset+'.pickle'), 'rb') as f:
            self.set = pickle.load(f)
        self.subset = subset
        self.datafile = os.path.join(self.dataset_folder, 'testing/'+subset+'_bev_'+RESOLUTION+'.npy')
        if MODE_ == 'height':
            self.datafile = os.path.join(self.dataset_folder, 'testing/'+subset+'_height_'+RESOLUTION+'.npy')
        print("loading bev file {}".format(self.datafile))

        self.dataset='wp'
        if not os.path.exists(self.datafile):
            generate_bevs(self.set,self.dataset_folder, self.datafile)
        if inputc == 1:
            self.bevs = np.load(self.datafile)[:,:,:,-1:]
            if USE_NORM:
                L, H, W, _ = self.bevs.shape
                for i in range(L):
                    sub_tensor = self.bevs[i, :]
                    self.bevs[i, :] = sub_tensor / np.max(sub_tensor)
        else:
            self.bevs = np.load(self.datafile) # 5->3
        self.pts_step =5

    def __len__(self):
        return len(self.set)

    def __getitem__(self, index):
        img = self.bevs[index]
        img = torch.tensor(img).permute(2,0,1) # CHW
        if self.input_transform:
            img = img.unsqueeze(0)
            img = self.input_transform(img)
            img = img.squeeze(0)
            return img,index
        else:
            img = img.repeat(3, 1, 1) 
            img = input_transform()(img)
            img *= 255
            xs, ys = np.meshgrid(np.arange(self.pts_step,img.size()[1]-self.pts_step,self.pts_step), np.arange(self.pts_step,img.size()[2]-self.pts_step,self.pts_step))
            xs=xs.reshape(-1,1)
            ys = ys.reshape(-1,1)
            pts = np.hstack((xs,ys))
            return (img,pts), index


class InterSet(Dataset):
    def __init__(self, inputc, img_size, set, subset):
        self.dataset_folder = root_dir
        self.positives = None
        if img_size:
            self.input_transform = transform(img_size)

        self.set = set
        self.datafile = os.path.join(self.dataset_folder, 'testing/'+subset+'_bev_'+RESOLUTION+'.npy')
        if MODE_ == 'height':
            self.datafile = os.path.join(self.dataset_folder, 'testing/'+subset+'_height_'+RESOLUTION+'.npy')
        print("loading bev file {}".format(self.datafile))

        self.dataset='wp'
        if not os.path.exists(self.datafile):
            generate_bevs(self.set,self.dataset_folder, self.datafile)
        if inputc == 1:
            self.bevs = np.load(self.datafile)[:,:,:,-1:]
            if USE_NORM:
                L, H, W, _ = self.bevs.shape
                for i in range(L):
                    sub_tensor = self.bevs[i, :]
                    self.bevs[i, :] = sub_tensor / np.max(sub_tensor)
        else:
            self.bevs = np.load(self.datafile) # 5->3

    def __len__(self):
        return len(self.set)

    def __getitem__(self, index):
        img = self.bevs[index]
        img = torch.tensor(img).permute(2,0,1) # CHW
        img = img.unsqueeze(0)
        if self.input_transform:
            img = self.input_transform(img)
            img = img.squeeze(0)
        return img, index

if __name__ == '__main__':
    pickle_path = '/A_dataset/Wild-Places/training/training_wild-places.pickle'
    with open(pickle_path, 'rb') as f:
        file = pickle.load(f)
    T = TrainDataset(file, '/A_dataset/Wild-Places/')
    a = T(10)