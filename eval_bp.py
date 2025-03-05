# coding=utf-8
'''
Author: shenyanqing1105 1159364090@qq.com
Date: 2025-03-05 15:51:09
LastEditors: shenyanqing1105 1159364090@qq.com
LastEditTime: 2025-03-05 15:52:53
FilePath: /ForestLPR-CVPR2025/eval_bp.py
'''
'''
BEVPlace testing
'''
from models.bp.bevplace import BEVPlace
import numpy as np
import argparse
from os.path import join, exists
import torch
from torch.utils.data import DataLoader
from util import Recall_at_N
import json
import faiss
import tqdm
from scripts.eval.utils import intra_Recall_N, inter_Recall_N
import h5py
import scripts.generate_splits.bevwp_match as bevwp
import pickle

try:
    from torchvision.models.utils import load_state_dict_from_url # torchvision 0.4+
except ModuleNotFoundError:
    try:
        from torch.hub import load_state_dict_from_url # torch 1.x
    except ModuleNotFoundError:
        from torch.utils.model_zoo import load_url as load_state_dict_from_url # torch 0.4.1

np.seterr(divide='ignore')

parser = argparse.ArgumentParser(description='BEVPlace')
parser.add_argument('--cacheBatchSize', type=int, default=8, help='Batch size for testing')
parser.add_argument('--nocuda', action='store_true', help='Dont use cuda')
parser.add_argument('--threads', type=int, default=40, help='Number of threads for each data loader to use')
parser.add_argument('--resume', type=str, default='./runs/checkpoint_paper_kitti.pth.tar', help='Path to load checkpoint from, for resuming training or testing.')
parser.add_argument('--img_size', type=int, default=None, nargs=2)
parser.add_argument('--features_dim', type=int, default=256)
parser.add_argument('--inputc', type=int, default=1, help='input channels')
parser.add_argument('--subset', type=str, default='V-04', help='which Intra/Inter dataset',choices=['None','3-02', '5-03', '5-06','V-01','V-02','V-03','V-04','K-01','K-02','K-03','K-04','Venman','Karawatha'])
parser.add_argument('--dataset', type=str, default='intra&inter&test&val')
parser.add_argument('--ngpu', type=int, default=1,
                    help='number of GPU to use.')
parser.add_argument('--gpu-id', type=int, default=0)
parser.add_argument('--root', type=str, default='/dataset/Wild-Places')
parser.add_argument('--plotPath', type=str, default=None,
                    help='Path for plot test result.')
parser.add_argument('--runsPath', type=str,
                    default='./runs/wildplaces/bev', help='Path to save runs to.')
parser.add_argument('--world_thresh', type = float, default = 3, help = 'Distance to be considered revisit in world')
parser.add_argument('--time_thresh', type = float, default = 600, help = 'Time before a previous frame can be considered a valid revisit')
parser.add_argument('--similarity_function', type = str, default = 'cosine', help = 'Distance function used to calculate similarity of embeddings')


def main():
    # =====> parse train flag
    opt = parser.parse_args()
    torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    print(opt)

# =====> build model
    print('===> Building model')
    model = BEVPlace()
    resume_ckpt = opt.resume

    print("=> loading checkpoint '{}'".format(resume_ckpt))
    checkpoint = torch.load(resume_ckpt, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'],strict=False)
    print("=> loaded checkpoint '{}' (epoch {})"
            .format(resume_ckpt, checkpoint['epoch']))


# =====> to device
    cuda = not opt.nocuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run with --nocuda")
    if cuda:
        torch.cuda.set_device(opt.gpu_id)
        device_ids = list(range(opt.ngpu))
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        model.cuda(opt.gpu_id)


# =====>define transform #
    if opt.img_size:
        opt.img_size = tuple(opt.img_size)
    else:
        opt.img_size = None
    print(opt.img_size)


# =====> load dataset
    if 'intra' in opt.dataset:
        print('====> Eval Intra-set')
        intra_set = bevwp.IntraSet(img_size=opt.img_size, inputc = opt.inputc, subset=opt.subset)
        recalls = validate(intra_set, model, opt)
        print(recalls)
    if 'inter' in opt.dataset:
        print('====> Eval Inter-set')
        recalls = inter_validate(opt, model)
        print(recalls)
    if 'test' in opt.dataset:
        print('====> Eval Test-set')
        whole_test_set = bevwp.TestSet(img_size=opt.img_size, inputc = opt.inputc)
        recalls = validate(whole_test_set, model, opt)
    if 'val' in opt.dataset:
        print('====> Eval Val-set')
        whole_val_set = bevwp.ValSet(img_size=opt.img_size, inputc = opt.inputc) # ! train
        recalls = validate(whole_val_set, model)

def inter_validate(opt, model):
    db_pickle_path = join(opt.root,'testing',opt.subset+'_evaluation_database.pickle')
    q_pickle_path = join(opt.root,'testing',opt.subset+'_evaluation_query.pickle')
    with open(q_pickle_path, 'rb') as f:
        query_sets = pickle.load(f) # L=4
    with open(db_pickle_path, 'rb') as f:
        database_sets = pickle.load(f)
    query_feat=[]
    database_feat=[]
    if opt.subset == 'Venman':
        folders=['V-01','V-02','V-03','V-04']
    elif opt.subset == 'Karawatha':
        folders=['K-01','K-02','K-03','K-04']

    for i in range(len(query_sets)):
        eval_set = bevwp.InterSet(img_size=opt.img_size, inputc = opt.inputc, set=query_sets[i], subset='Q'+folders[i])
        test_data_loader = DataLoader(dataset=eval_set,
                                    num_workers=opt.threads,
                                    batch_size=opt.cacheBatchSize,
                                    shuffle=False, pin_memory=True)
        dbFeat = np.empty((len(eval_set), opt.features_dim))
        with torch.no_grad():
            for iteration, (input, indices) in tqdm.tqdm(enumerate(test_data_loader, 1), desc='Computing Features', total=len(eval_set)/opt.cacheBatchSize):
                if (not opt.nocuda) and torch.cuda.is_available():
                    input = to_cuda(input)
                encoding = model(input)
                dbFeat[indices.detach().numpy(), :] = encoding.detach().cpu().numpy()
                del input, encoding
            query_feat.append(dbFeat.astype('float32'))
            del dbFeat
    for i in range(len(database_sets)):
        eval_set = bevwp.InterSet(img_size=opt.img_size, inputc = opt.inputc, set=database_sets[i], subset='DB'+folders[i])
        test_data_loader = DataLoader(dataset=eval_set,
                                    num_workers=opt.threads,
                                    batch_size=opt.cacheBatchSize,
                                    shuffle=False, pin_memory=True)
        dbFeat = np.empty((len(eval_set), opt.features_dim))
        with torch.no_grad():
            for iteration, (input, indices) in tqdm.tqdm(enumerate(test_data_loader, 1), desc='Computing Features', total=len(eval_set)/opt.cacheBatchSize):
                if (not opt.nocuda) and torch.cuda.is_available():
                    input = to_cuda(input)
                encoding = model(input)
                dbFeat[indices.detach().numpy(), :] = encoding.detach().cpu().numpy()
                del input, encoding
            database_feat.append(dbFeat.astype('float32'))
            del dbFeat
    recalls = inter_Recall_N(query_sets, database_sets, query_feat, database_feat, opt.subset, opt)
    return recalls

def validate(eval_set, model, args, epoch=None, write_TBoard=False, writer=None,mode='val'):
    feature_dim = args.features_dim
    test_data_loader = DataLoader(dataset=eval_set,
                                  num_workers=args.threads,
                                  batch_size=args.cacheBatchSize,
                                  shuffle=False, pin_memory=True)
    dbFeat = np.empty((len(eval_set), feature_dim))
    model.eval()
    cachefile = './data/{}_{}_{}_feat_cache.hdf5'.format(args.resume.split('/')[-3], args.dataset, args.subset)
    if exists(cachefile):
        print('loading exist featcache file:{}'.format(cachefile))
        with h5py.File(cachefile, mode='r+') as h5:
            h5Feat = h5.get('features')
            dbFeat=h5Feat[:]
            qFeat=h5Feat[:]
    else:
        with torch.no_grad():
            for iteration, (input, indices) in tqdm.tqdm(enumerate(test_data_loader, 1), desc='Computing Features', total=len(eval_set)/args.cacheBatchSize):
                if (not args.nocuda) and torch.cuda.is_available():
                    input = to_cuda(input)
                encoding = model(input)
                dbFeat[indices.detach().numpy(), :] = encoding.detach().cpu().numpy()
                del input, encoding
            qFeat = dbFeat.astype('float32')
            dbFeat = dbFeat.astype('float32')

            with h5py.File(cachefile, mode='w') as h5:  # ? 放上面?
                h5feat = h5.create_dataset("features",
                                        [len(eval_set), feature_dim], dtype=np.float32)
                h5feat[:]=dbFeat

    n_values = [1, 5, 10, 20, 100]
    print('====> Building faiss index')
    faiss_index = faiss.IndexFlatL2(feature_dim)
    faiss_index.add(dbFeat)

    print('====> Calculating recall @ N [1,5,10,20,100]')
    if eval_set.subset == 'train_val' or eval_set.subset == 'test':
        _, predictions = faiss_index.search(
            qFeat, max(n_values))

        gt, values = eval_set.getPositives()
        for qIx, pred in enumerate(predictions):
            indexes = np.where(np.isin(pred, values[qIx]))[0]
            rest_elements = np.delete(pred,indexes)
            moved_elements = np.intersect1d(pred,values[qIx])
            if len(rest_elements)>0:
                predictions[qIx,:] = np.append(rest_elements, moved_elements)
            else:
                gt[qIx] = predictions[qIx].tolist()
        metrics = Recall_at_N(predictions,gt,epoch,write_TBoard,writer,mode=mode)
    else:
        metrics = intra_Recall_N(eval_set.set,dbFeat, args)
    return metrics

def to_cuda(data):
    results = []
    for i, item in enumerate(data):
        if type(item).__name__ == "Tensor":
            results.append(item.cuda())
        elif type(item).__name__ == 'list':
            tensor_list = []
            for tensor in item:
                if type(tensor).__name__ == "Tensor":
                    tensor_list.append(tensor.cuda())
                else:
                    tensor_list2 = []
                    for tensor_i in tensor:
                        tensor_list2.append(tensor_i.cuda())
                    tensor_list.append(tensor_list2)
            results.append(tensor_list)
        else:
            raise NotImplementedError
    return results

if __name__ == '__main__':
    main()
    # sys.stdout.close()
    # sys.stdout=stdoutOrigin