# coding=utf-8
'''
stable version
'''
from time import time
# from models.model.Deit import deit_small_distilled_patch16_224, deit_base_distilled_patch16_384
from models.model.Deit_multi import deit_small_distilled_patch16_224, deit_base_distilled_patch16_384
import math
import numpy as np
import argparse
from os import makedirs
from os.path import join, exists, isfile
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from tensorboardX import SummaryWriter
from datetime import datetime
from math import ceil
import random
import json
import faiss
from utils import Recall_at_N, save_checkpoint

import scripts.generate_splits.bevwp as bevwp

try:
    from torchvision.models.utils import load_state_dict_from_url # torchvision 0.4+
except ModuleNotFoundError:
    try:
        from torch.hub import load_state_dict_from_url # torch 1.x
    except ModuleNotFoundError:
        from torch.utils.model_zoo import load_url as load_state_dict_from_url # torch 0.4.1


np.seterr(divide='ignore')


parser = argparse.ArgumentParser(description='BEVPR')
parser.add_argument('--img_size', type=int, default=None, nargs=2)
parser.add_argument('--level', type=int, default=[1,6,11], nargs=3)
parser.add_argument('--depth', type=int, default=6)
parser.add_argument('--features_dim', type=int, default=256)
parser.add_argument('--inputc', type=int, default=1)
parser.add_argument('--batchSize', type=int, default=2,
                    help='Number of triplets (query, pos, negs). Each triplet consists of 12 images.')
parser.add_argument('--cacheBatchSize', type=int, default=24,
                    help='Batch size for caching and testing')
parser.add_argument('--cacheRefreshRate', type=int, default=1000,
                    help='How often to refresh cache, in number of queries. 0 for off')
parser.add_argument('--nEpochs', type=int, default=100,
                    help='number of epochs to train for')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--ngpu', type=int, default=1,
                    help='number of GPU to use.')
parser.add_argument('--gpu-id', type=int, default=0)
parser.add_argument('--agg', type=str, default='gem',
                    help='aggregation layer', choices=['gem', 'cls'])
parser.add_argument('--optim', type=str, default='ADAMW',
                    help='optimizer to use', choices=['SGD', 'ADAM', 'ADAMW'])
parser.add_argument('--warmup', default=3, type=int,
                    help='number of warmup epochs')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='Learning Rate.')
parser.add_argument('--lrStep', type=float, default=5,
                    help='Decay LR ever N steps.')
parser.add_argument('--lrGamma', type=float, default=0.5,
                    help='Multiply LR by Gamma for decaying.')
parser.add_argument('--weightDecay', type=float,
                    default=0.0001, help='Weight decay for SGD.')
parser.add_argument('--momentum', type=float,
                    default=0.9, help='Momentum for SGD.')
parser.add_argument('--nocuda', action='store_true', help='Dont use cuda')
parser.add_argument('--threads', type=int, default=8,
                    help='Number of threads for each data loader to use')
parser.add_argument('--seed', type=int, default=123,
                    help='Random seed to use.')
parser.add_argument('--plotPath', type=str, default='./plot/rgb',
                    help='Path for plot test result.')
parser.add_argument('--dataPath', type=str, default='./data/rgb',
                    help='Path for centroid data.')
parser.add_argument('--runsPath', type=str,
                    default='./runs/wildplaces/bev', help='Path to save runs to.')
parser.add_argument('--savePath', type=str, default='checkpoints',
                    help='Path to save checkpoints to in logdir. Default=checkpoints/')
parser.add_argument('--cachePath', type=str, default='./tmp',
                    help='Path to save cache to.')
parser.add_argument('--resume', type=str, default='',
                    help='Path to load checkpoint from, for resuming training or testing. .tar file')
parser.add_argument('--evalEvery', type=int, default=1,
                    help='Do a validation set run, and save, every N epochs.')
parser.add_argument('--patience', type=int, default=30,
                    help='Patience for early stopping. 0 is off.')
parser.add_argument('--arch', type=str, default='deit-s',
                    help='basenetwork to use', choices=['deit-s', 'deit-b'])
parser.add_argument('--margin', type=float, default=0.1,
                    help='Margin for triplet loss. Default=0.1')
parser.add_argument('--metric', type=str, default='RN', help='which metric is used',
                    choices=['AUC_RN', 'AUC', 'RN'])
parser.add_argument('--pretrained', action='store_true',
                    help='if use pretrained model')
parser.add_argument('--firsteval', action='store_true')


def main():
    # =====> parse train flag
    opt = parser.parse_args()

    restore_var = ['lr', 'lrStep', 'lrGamma', 'weightDecay', 'momentum', 'warmup',
                   'runsPath', 'savePath', 'optim', 'seed']
    if opt.resume:
        flag_file = join(opt.resume, 'checkpoints', 'flags.json')
        if exists(flag_file):  # isfile
            print(">> Loading train flags:\n>> '{}'".format(flag_file))

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
                print('Restored flags:', train_flags)
                # update
                opt = parser.parse_args(train_flags, namespace=opt)
        else:
            print(">> No flags.json found at '{}'".format(flag_file))
            print(opt.runsPath)
    print(opt)

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)

    torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
# =====> build model
    print('===> Building model')
    if opt.arch == 'deit-s':
        model = deit_small_distilled_patch16_224(img_size=opt.img_size, num_classes=opt.features_dim, inputc = opt.inputc, depth=opt.depth,agg=opt.agg, level=opt.level)
    if opt.arch == 'deit-b':
        model = deit_base_distilled_patch16_384(img_size=opt.img_size, num_classes=opt.features_dim, inputc = opt.inputc, depth=opt.depth,agg=opt.agg, level=opt.level)
    print('dim:', opt.features_dim)
# =====> optimizer
    if opt.optim.upper() == 'ADAM':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                      model.parameters()), lr=opt.lr)
    elif opt.optim.upper() == 'ADAMW':
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad,
                                       model.parameters()),
                                lr=opt.lr, betas=(0.9, 0.99),
                                weight_decay=opt.weightDecay)
    elif opt.optim.upper() == 'SGD':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad,
                                     model.parameters()),
                              lr=opt.lr, momentum=opt.momentum,
                              weight_decay=opt.weightDecay)
        scheduler = optim.lr_scheduler.StepLR(optimizer,
                                              step_size=opt.lrStep,
                                              gamma=opt.lrGamma)
    else:
        raise ValueError('Unknown optimizer: ' + opt.optim)


# =====> loss
    criterion = nn.TripletMarginLoss(
        margin=opt.margin, p=2, reduction='sum')
    criterion_cos = nn.CosineSimilarity(dim=-1)

# =====> model param

    if opt.resume:
        resume_ckpt = opt.resume

        if isfile(resume_ckpt):
            loc = 'cuda:{}'.format(opt.gpu_id)
            checkpoint = torch.load(
                resume_ckpt, map_location=loc)  # , map_location=lambda storage, loc: storage)
            # todo
            opt.start_epoch = checkpoint['epoch']  # default start_epoch = 0
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            if opt.optim.upper() == 'SGD':
                scheduler = optim.lr_scheduler.StepLR(optimizer,
                                                      step_size=opt.lrStep, gamma=opt.lrGamma,
                                                      last_epoch=checkpoint['epoch']-1)  # -1
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
        criterion = criterion.cuda(opt.gpu_id)
        criterion_cos = criterion_cos.cuda(opt.gpu_id)
        torch.cuda.manual_seed(opt.seed)
        torch.cuda.manual_seed_all(opt.seed)


# =====>define transform
    if opt.img_size:
        opt.img_size = tuple(opt.img_size)
    else:
        opt.img_size = (360, 1440)
    print(opt.img_size)


# =====> load dataset
    opt.dataset = 'wp'
    print('===> Loading wild-places {} dataset(s)'.format('train'))
    train_set = bevwp.TrainDataset(opt.img_size, nNeg=3, margin = opt.margin, inputc = opt.inputc)
    print('====> Training query set:', len(train_set))
    whole_train_set = bevwp.ValSet(img_size=opt.img_size, inputc = opt.inputc) # update feat
    whole_train_set.bevs=train_set.bevs
    whole_training_data_loader = DataLoader(dataset=whole_train_set, 
            num_workers=opt.threads, batch_size=opt.cacheBatchSize, shuffle=False, 
            pin_memory=cuda)

    whole_test_set = bevwp.TestSet(img_size=opt.img_size, inputc = opt.inputc)
    whole_val_set = whole_train_set
    print('===> Loading wild-places {} dataset(s)'.format('cache'))

    if not exists(opt.dataPath):
        makedirs(opt.dataPath)
    train_set.Featcache = join(
        opt.dataPath, opt.arch+'_'+str(opt.batchSize)+'_'+datetime.now().strftime('%b%d_%H-%M-%S')+'_train_feat_cache.hdf5')



# =====> runing
    print('====> Training model')
    # summary writer
    writer = SummaryWriter(log_dir=join(
        opt.runsPath, datetime.now().strftime('%b%d_%H-%M-%S')
        + '_'+opt.arch))
    dataiter = iter(whole_training_data_loader)
    input, indices = next(dataiter)
    input = input.cuda(opt.gpu_id, non_blocking=True)
    writer.add_graph(model.module, input)

    logdir = writer.file_writer.get_logdir()
    savePath = join(logdir, opt.savePath)
    makedirs(savePath)

    with open(join(savePath, 'flags.json'), 'w') as f:
        f.write(json.dumps(vars(opt))) ## store all params

    print('===> Saving state to:', logdir)
    if opt.firsteval:
        print('====> First Eval Test-set')
        recalls = validate(whole_test_set, model, opt, opt.start_epoch,
                            write_TBoard=True, writer=writer,mode='test')
        print('====> First Eval Val-set')
        recalls = validate(whole_val_set, model, opt, opt.start_epoch,
                            write_TBoard=True, writer=writer,mode='val')

    not_improved = 0
    best_score = 0
    for epoch in range(opt.start_epoch+1, opt.nEpochs+1):
        # from 1
        time_begin = time()
        adjust_learning_rate(optimizer, epoch, opt, writer)
        train(train_set, whole_train_set, whole_training_data_loader,
              model, criterion, criterion_cos, optimizer, epoch, opt, writer)
        if opt.optim.upper() == 'SGD':
            scheduler.step(epoch)  # adjust learning rate for each epoch
        total_mins = (time() - time_begin) / 60
        print(f'[Epoch {epoch}] \t  Time: {total_mins:.2f}')

        if (epoch % opt.evalEvery) == 0:
            print('====> Eval Val-set')
            recalls = validate(whole_test_set, model, opt, epoch,
                               write_TBoard=True, writer=writer,mode='test')  # validate loss + metric
            recalls = validate(whole_val_set, model, opt, epoch,
                               write_TBoard=True, writer=writer,mode='val')  # validate loss + metric
            is_best = recalls[1] > best_score
            if is_best:
                not_improved = 0
                best_score = recalls[1]
            else:
                not_improved += 1
            if is_best: # only save the best
                save_checkpoint({'epoch': epoch,
                                'state_dict': model.state_dict(),
                                'recalls': recalls,
                                'best_score': best_score,
                                'optimizer': optimizer.state_dict()
                                }, is_best, savePath)

            if opt.patience > 0 and not_improved > (opt.patience/opt.evalEvery):
                print('Performance did not improve for',
                      opt.patience, 'epochs. Stopping.')
                break
    print("=> Best Recall@5: {:.4f}".format(best_score), flush=True)
    writer.close()


def train(train_set, whole_train_set, whole_training_data_loader, model, criterion, criterion_cos, optimizer, epoch, args, writer=None):
    if not exists(args.cachePath):
        makedirs(args.cachePath)

    with h5py.File(train_set.Featcache, mode='w') as h5:
        feature_dim = args.features_dim
        h5feat = h5.create_dataset("features",
                                [len(whole_train_set), feature_dim], dtype=np.float32)
    epoch_loss = 0
    startIter = 1

    if args.cacheRefreshRate > 0:
        subsetN = ceil(len(train_set)/args.cacheRefreshRate)
        subsetIdx = np.array_split(np.arange(len(train_set)), subsetN)
    else:
        subsetN = 1
        subsetIdx = [np.arange(len(train_set))]

    nBatches = (len(train_set) + args.batchSize -
                1)//args.batchSize


    for subIter in range(subsetN):
        print('====> Building Cache')
        model.eval()
        with torch.no_grad():
            with h5py.File(train_set.Featcache, mode='r+') as h5:
                h5feat = h5.get('features')
                for iteration, (input, indices) in enumerate(whole_training_data_loader):
                    input = input.cuda(args.gpu_id, non_blocking=True)
                    trans_encoding = model(input)
                    h5feat[indices.detach().numpy(),
                           :] = trans_encoding.detach().cpu().numpy()
                    del input, trans_encoding

        print('====> Refresh train set cache')
        print('====> Continue Training')

        # train
        sub_train_set = Subset(dataset=train_set, indices=subsetIdx[subIter]) # for refresh

        training_data_loader = DataLoader(dataset=sub_train_set,
                                          num_workers=args.threads,
                                          batch_size=args.batchSize,
                                          shuffle=True,
                                          collate_fn=bevwp.collate_fn,
                                          pin_memory=True)

        model.train()
        for iteration, (query, positive, negatives,
                        negCounts, indices) in enumerate(training_data_loader,
                                                         startIter):

            if query is None:
                continue
            B, C, H, W = query.shape
            nNeg = torch.sum(negCounts)

            input = torch.cat([query, positive, negatives])
            input = input.cuda(args.gpu_id, non_blocking=True)

            trans_encoding = model(input)

            transQ, transP, transN = torch.split(
                trans_encoding, [B, B, nNeg])
            optimizer.zero_grad()

            loss = 0

            for i, negCount in enumerate(negCounts):
                for n in range(negCount):
                    negIx = (torch.sum(negCounts[:i])+n).item()
                    loss += criterion(transQ[i:i+1],
                                      transP[i:i+1], transN[negIx:negIx+1])

            loss /= nNeg.float().cuda(args.gpu_id, non_blocking=True)

            loss.backward()
            optimizer.step()

            del input, trans_encoding, transQ, transP, transN
            del query, positive, negatives

            batch_loss = loss.item()
            epoch_loss += batch_loss
            del loss
            if iteration % 50 == 0 or nBatches <= 10:
                print('==> Epoch[{}]({}/{}): Loss: {:.4f}'.
                      format(epoch, iteration, nBatches, batch_loss),
                      flush=True)
                writer.add_scalar('Train/Loss', batch_loss,
                                  ((epoch-1)*nBatches)+iteration)
                writer.add_scalar('Train/nNeg', nNeg,
                                  ((epoch-1)*nBatches)+iteration)

        startIter += len(training_data_loader)
        del training_data_loader
        optimizer.zero_grad()

        torch.cuda.empty_cache()

    avg_loss = epoch_loss / nBatches
    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, avg_loss),
          flush=True)
    writer.add_scalar('Train/AvgLoss', avg_loss, epoch)


def validate(eval_set, model, args, epoch=None, write_TBoard=False, writer=None,mode='val'):

    feature_dim = args.features_dim
    test_data_loader = DataLoader(dataset=eval_set,
                                  num_workers=args.threads,
                                  batch_size=args.cacheBatchSize,
                                  shuffle=False, pin_memory=True)
    dbFeat = np.empty((len(eval_set), feature_dim))

    model.eval()
    with torch.no_grad():
        for iteration, (input, indices) in enumerate(test_data_loader, 1):
            if (not args.nocuda) and torch.cuda.is_available():
                input = input.cuda(args.gpu_id, non_blocking=True)
            encoding = model(input)
            dbFeat[indices.detach().numpy(), :] = encoding.detach().cpu().numpy()
            del input, encoding
        qFeat = dbFeat.astype('float32')
        dbFeat = dbFeat.astype('float32')

    if 'RN' in args.metric:
        n_values = [1, 5, 10, 20, 100]
        print('====> Building faiss index')
        faiss_index = faiss.IndexFlatL2(feature_dim)
        faiss_index.add(dbFeat)

        print('====> Calculating recall @ N [1,5,10,20,100]')

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
        return metrics


def adjust_learning_rate(optimizer, epoch, args, writer):
    lr = args.lr
    if hasattr(args, 'warmup') and epoch < args.warmup:
        lr = lr / (args.warmup - epoch)
    else:
        alpha = 0.1
        cosine_decay = 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup) / (args.nEpochs - args.warmup)))
        decayed = (1 - alpha) * cosine_decay + alpha
        lr *= decayed
        writer.add_scalar('param/lr',
                          lr, epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()