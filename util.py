import numpy as np 
import pandas as pd 
import shapely.affinity
from shapely.geometry import Polygon, Point
import torch
from os.path import join
import shutil
# Venman
P1 = Polygon([(-468,-82), (-468,44), (-314,44), (-305,12), (-192,44), (-192,-82)])
P2 = Polygon([(-78,-171), (-78,-215), (-305,-215), (-305,-171)])
P3 = Polygon([(-62, 70), (95, 70), (142, 0), (140, -142), (-62, -142)])

# Karawatha
P4 = Polygon([(-150, 8), (300,8), (300,-210), (-150,-210)])
P5 = Polygon([(-215,618), (-74,618), (-74,423), (-215,423)])
P6 = Polygon([(-513,300), (-513,37), (-321,37), (-321,300)])

def make_circle(x,y, radius = 30):
    circle = Point(x,y).buffer(1) 
    circle = shapely.affinity.scale(circle, radius, radius)
    return circle

# Venman
B1 = make_circle(-63, 40)
B2 = make_circle(114,-143)
B3 = make_circle(-77,-205)
B4 = make_circle(-310,-171)
B5 = make_circle(-433,-82)
B6 = make_circle(-189,12)

# Karawatha
B7 = make_circle(-216,606)
B8 = make_circle(-98,428)
B9 = make_circle(-316,260)
B10 = make_circle(-321,63)
B11 = make_circle(-149,-22)
B12 = make_circle(300,-134)

def load_csv(csv_path, rel_cloud_path):
    df = pd.read_csv(csv_path, delimiter = ',', dtype = str)
    df = df.astype({'x': float, 'y':float, 'z':float, 'qx':float, 'qy':float, 'qz':float, 'qw':float, 'timestamp': str})
    df['easting'] = df['x']
    df['northing'] = df['y']
    df['filename'] = rel_cloud_path + '/' + df['timestamp'] + '.pcd'
    df = df[['filename', 'northing', 'easting', 'x','y','z','qx','qy','qz','qw']]
    return df 

def check_in_test_set(easting, northing, test_polygons, exclude_polygons):
    split = 'train'
    point = Point(easting, northing)
    for poly in test_polygons:
        if poly.contains(point):
            split = 'test'
            return split 
    for poly in exclude_polygons:
        if poly.contains(point):
            split = 'buffer'
            return split 
    return split  


class TrainingTuple:
    # Tuple describing an element for training/validation
    def __init__(self, id: int, timestamp: int, rel_scan_filepath: str, positives: np.ndarray, positives_conf: np.ndarray, positives_dist: np.ndarray,
                 non_negatives: np.ndarray, position: np.ndarray, pose = None):
        # id: element id (ids start from 0 and are consecutive numbers)
        # ts: timestamp
        # rel_scan_filepath: relative path to the scan
        # positives: sorted ndarray of positive elements [id]
        # negatives: sorted ndarray of elements id
        # position: x, y position in meters (northing, easting)
        assert position.shape == (2,)

        self.id = id
        self.timestamp = timestamp
        self.rel_scan_filepath = rel_scan_filepath
        self.positives = positives
        self.positives_conf = positives_conf
        self.positives_dist = positives_dist
        self.non_negatives = non_negatives # potential positives
        self.position = position
        self.pose = pose

def save_checkpoint(state, is_best, savePath, filename='checkpoint.pth.tar'):
    filename = 'model_epoch%d.pth.tar' % state['epoch']
    model_out_path = join(savePath, filename)
    torch.save(state, model_out_path)
    if is_best:
        shutil.copyfile(model_out_path, join(
            savePath, 'model_best.pth.tar'))


def Recall_at_N(predictions, gt, epoch=0, write_TBoard=False, writer=None, mode = 'val', NMS=False):
    '''
    predictions : rank index matrix for queries according to the similarity
    gt : list of list
    write_TBoard : validate(True) test(False)
    '''
    n_values = [1, 5, 10, 15, 20, 100]
    correct_at_n = np.zeros(len(n_values))
    for qIx, pred in enumerate(predictions):
        for i, n in enumerate(n_values):
            if np.any(np.in1d(pred[:n], gt[qIx])):
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
        if write_TBoard:
            writer.add_scalar(mode+'/Recall@' + str(n),
                              recall_at_n[i], epoch)

    return recalls
