# from models.model.Deit import deit_small_distilled_patch16_224
from models.model.Deit_multi import deit_small_distilled_patch16_224
import numpy as np
import argparse
from os.path import join, exists, isfile
import torch
from torch.utils.data import DataLoader
from util import Recall_at_N
import json
import faiss
import tqdm
from scripts.eval.utils import intra_Recall_N, inter_Recall_N
import h5py
import scripts.generate_splits.bevwp as bevwp
import pickle

try:
    from torchvision.models.utils import load_state_dict_from_url # torchvision 0.4+
except ModuleNotFoundError:
    try:
        from torch.hub import load_state_dict_from_url # torch 1.x
    except ModuleNotFoundError:
        from torch.utils.model_zoo import load_url as load_state_dict_from_url # torch 0.4.1


model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}

np.seterr(divide='ignore')


parser = argparse.ArgumentParser(description='BEVPR')
parser.add_argument('--img_size', type=int, default=None, nargs=2)
parser.add_argument('--level', type=int, default=[1,6,11], nargs=3)
parser.add_argument('--depth', type=int, default=6)
parser.add_argument('--features_dim', type=int, default=256)
parser.add_argument('--inputc', type=int, default=1)
parser.add_argument('--subset', type=str, default='V-04', help='which Intra dataset')
parser.add_argument('--dataset', type=str, default='intra&test&val')
parser.add_argument('--cacheBatchSize', type=int, default=24,
                    help='Batch size for caching and testing')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--ngpu', type=int, default=1,
                    help='number of GPU to use.')
parser.add_argument('--gpu-id', type=int, default=0)
parser.add_argument('--agg', type=str, default='gem',
                    help='aggregation layer', choices=['gem', 'cls'])
parser.add_argument('--nocuda', action='store_true', help='Dont use cuda')
parser.add_argument('--root', type=str, default='/A_dataset/Wild-Places')
parser.add_argument('--plotPath', type=str, default=None,
                    help='Path for plot test result.')
parser.add_argument('--runsPath', type=str,
                    default='./runs/wildplaces/bev', help='Path to save runs to.')
parser.add_argument('--resume', type=str, default='',
                    help='Path to load checkpoint from, for resuming training or testing.')
parser.add_argument('--evalEvery', type=int, default=1,
                    help='Do a validation set run, and save, every N epochs.')
parser.add_argument('--arch', type=str, default='deit-s',
                    help='basenetwork to use', choices=['deit-s', 'deit-b'])
parser.add_argument('--metric', type=str, default='RN', help='which metric is used',
                    choices=['AUC_RN', 'AUC', 'RN'])
parser.add_argument('--threads', type=int, default=8,
                    help='Number of threads for each data loader to use')
parser.add_argument('--world_thresh', type = float, default = 3, help = 'Distance to be considered revisit in world')
parser.add_argument('--time_thresh', type = float, default = 600, help = 'Time before a previous frame can be considered a valid revisit')
parser.add_argument('--similarity_function', type = str, default = 'cosine', help = 'Distance function used to calculate similarity of embeddings')

def main():
    # =====> parse train flag
    opt = parser.parse_args()
    torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    if opt.resume:
        restore_var = ['agg', 'features_dim', 'depth', 'inputc']
        flag_file = join(opt.resume, 'flags.json')
        if exists(flag_file):  # isfile
            print(">> Loading ckpt flags:\n>> '{}'".format(flag_file))

            with open(flag_file, 'r') as f:
                stored_flags = {
                    '--'+k: str(v) for k, v in json.load(f).items() if k in restore_var}
                to_del = []
                for flag, val in stored_flags.items():
                    for act in parser._actions:
                        if act.dest == flag[2:]:
                            if type(act.const) == type(True):
                                if val == str(act.default):
                                    to_del.append(flag)
                                else:
                                    stored_flags[flag] = ''
                for flag in to_del:
                    del stored_flags[flag]
                train_flags = [x for x in list(
                    sum(stored_flags.items(), tuple())) if len(x) > 0]
                print('Loaded flags:', train_flags)
                # update
                opt = parser.parse_args(train_flags, namespace=opt)
    print(opt)

# =====> build model
    print('===> Building model')
    model = deit_small_distilled_patch16_224(img_size=opt.img_size, num_classes=opt.features_dim, inputc = opt.inputc, depth=opt.depth,agg=opt.agg, level=opt.level)
    print('dim:', opt.features_dim)

    # optional load state_dict from checkpoint to update sth.
    resume_ckpt = join(opt.resume,'model_best.pth.tar')

    if isfile(resume_ckpt):
        loc = 'cuda:{}'.format(opt.gpu_id)
        checkpoint = torch.load(
            resume_ckpt, map_location=loc)  # , map_location=lambda storage, loc: storage)
        # todo
        opt.start_epoch = checkpoint['epoch']  # default start_epoch = 0
        model.load_state_dict(checkpoint['state_dict'], strict=False)

        print("=> loaded checkpoint '{}' (epoch {})"
                .format(resume_ckpt, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(resume_ckpt))


# =====> to device
    cuda = not opt.nocuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run with --nocuda")
    if cuda:
        torch.cuda.set_device(opt.gpu_id)
        device_ids = list(range(opt.ngpu))
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        model.cuda(opt.gpu_id)


# =====>define transform
    if opt.img_size:
        opt.img_size = tuple(opt.img_size)
    else:
        opt.img_size = (360, 1440)
    print(opt.img_size)

# =====> load dataset
    if 'intra' in opt.dataset:
        print('====> Eval Intra-set')
        intra_set = bevwp.IntraSet(img_size=opt.img_size, inputc = opt.inputc, subset=opt.subset)
        recalls = validate(intra_set, model, opt, opt.start_epoch)
        print(recalls)
    if 'inter' in opt.dataset:
        print('====> Eval Inter-set')
        recalls = inter_validate(opt, model)
        print(recalls)
    if 'test' in opt.dataset:
        print('====> Eval Test-set')
        whole_test_set = bevwp.TestSet(img_size=opt.img_size, inputc = opt.inputc)
        recalls = validate(whole_test_set, model, opt, opt.start_epoch)

    if 'val' in opt.dataset:
        print('====> Eval Val-set')
        whole_val_set = bevwp.ValSet(img_size=opt.img_size, inputc = opt.inputc)
        recalls = validate(whole_val_set, model, opt, opt.start_epoch)

def inter_validate(opt, model):
    db_pickle_path = join(opt.root,'testing',opt.subset+'_evaluation_database.pickle')
    q_pickle_path = join(opt.root,'testing',opt.subset+'_evaluation_query.pickle')
    with open(q_pickle_path, 'rb') as f:
        query_sets = pickle.load(f) # L=4
    with open(db_pickle_path, 'rb') as f:
        database_sets = pickle.load(f)
    query_feat=[]
    database_feat=[]
    if 'Venman' in opt.subset:
        folders=['V-01','V-02','V-03','V-04']
    elif 'Karawatha' in opt.subset:
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
                    input = input.cuda(opt.gpu_id, non_blocking=True)
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
                    input = input.cuda(opt.gpu_id, non_blocking=True)
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

    cachefile = './data/{}_{}_feat_cache.hdf5'.format(args.resume.split('/')[-2], args.subset)
    if exists(cachefile):
        print('loading exist featcache file')
        with h5py.File(cachefile, mode='r+') as h5:
            h5Feat = h5.get('features')
            dbFeat=h5Feat[:]
            qFeat=h5Feat[:]
    else:
        with torch.no_grad():
            for iteration, (input, indices) in tqdm.tqdm(enumerate(test_data_loader, 1), desc='Computing Features', total=len(eval_set)/args.cacheBatchSize):
                if (not args.nocuda) and torch.cuda.is_available():
                    input = input.cuda(args.gpu_id, non_blocking=True)
                encoding = model(input)
                dbFeat[indices.detach().numpy(), :] = encoding.detach().cpu().numpy()
                del input, encoding
            qFeat = dbFeat.astype('float32')
            dbFeat = dbFeat.astype('float32')

            with h5py.File(cachefile, mode='w') as h5:
                h5feat = h5.create_dataset("features",
                                        [len(eval_set), feature_dim], dtype=np.float32)
                h5feat[:]=dbFeat

    if 'RN' in args.metric:
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


if __name__ == '__main__':
    main()