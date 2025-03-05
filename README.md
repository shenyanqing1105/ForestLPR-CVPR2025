# ForestLPR: LiDAR Place Recognition in Forests Attentioning Multiple BEV Density Images

The code is almost there. Don't ask, we don't know either. Good luck!😉

source code of our CVPR'25 paper [ForestLPR: LiDAR Place Recognition in Forests Attentioning Multiple BEV Density Images]()

<img src="https://github.com/user-attachments/assets/21c49ddd-6f88-4237-bfe9-ef78bcd0e39c" width="400px">



## Table of Contents
[TOC]

## Environments

- CUDA 11.3
- Python 3.8.5
- PyTorch 1.10.2

We used Anaconda to setup a deep learning workspace that supports PyTorch. Run the following script to install the required packages.

```shell
[todo] conda env create -f environment.yml
conda create --name forestlpr_env python=3.8.5
conda activate forestlpr_env
git clone git@github.com:shenyanqing1105/ForestLPR-CVPR2025.git
cd ForestLPR-CVPR2025
pip install -r requirements.txt
conda deactivate
```

## Required Submap Data

We use two public datasets and one own-collected dataset: [Wild-Places](https://github.com/csiro-robotics/Wild-Places), [Botanic Garden Dataset](https://github.com/robot-pesg/BotanicGarden), and [ANYmal](). For [Botanic](), we repurpose the dataset and use GT to register multiple frames to 1 submap. These submaps should be placed in `Clouds_downsampled`

Additionally, in our work, we have processed the datasets into cropped point clouds [Cloud_normalized]() and BEV density images [download](), which is more convenient to test the BEV-based methods.

All of the datasets are placed  in `$HOME/ForestLPR/data`.

Environment visualization
![environments](https://github.com/user-attachments/assets/e799831c-b655-49e8-b4dd-0c4ca00539dc)

## dataset structure
```
data/
├── anymal/		  # Dataset
|   ├── anymal-3/ # Environment
|	│   └── 02/   # Sequence
|	|	    ├── Clouds_downsampled/
|	|	    ├── Clouds_normalized/
|	|	    └── poses_aligned.csv
|   └── testing/
|	    ├── 3-02.pickle
|	    └── npy files
├── botanic/    # Dataset
|   ├── 1005-03/  # Environment
|	│   └── 02/   # Sequence
|	|	    ├── Clouds_downsampled/
|	|	    ├── Clouds_normalized/
|	|	    └── poses_aligned.csv
|   ├── 1005-06/
|     └── ...
|   └── testing/
|	    ├── 5-03.pickle
|	    ├── 5-06.pickle
|	    └── npy files
└── Wild-Places/
    ├── Karawatha/
    |	├── K-01/
    |	|	├── Clouds_downsampled/
    |	|	└── poses_aligned.csv
    |	├── K-02/
    |   |	├── Clouds_downsampled/
    |	|	└── poses_aligned.csv
    |	├── K-03/
    |	|	├── Clouds_downsampled/
    |	|	└── poses_aligned.csv
    |	└── K-04/
    |    	├── Clouds_downsampled/
    |		└── poses_aligned.csv
    ├── testing/
    |	├── V-03.pickle
    |	├── V-04.pickle
    |	├── K-03.pickle
    |	├── V-03.pickle
    |	└── npy files
    ├── training/
    |	├── training_wild-places.pickle
    |	├── testing_wild-places.pickle
    |	└── npy files
    └── Venman/
            ├── V-01/
            ├── V-02/
            ├── V-03/
            └── V-04/
```
## Codebase

Here we provide the Code of Baselines for testing Wild-Places. todo align

## File structure




## Scripts
First check dataset file path in bevwp.py
### Pre-processing point clouds
which file, how to run, how to set the hpyer-parameters

### generating pickle files for each sequence

##### __Training__ - wildplaces

To generate the training splits run the following command:

```
python generate_splits/training_sets.py --dataset_root $_PATH_TO_DATASET --save_folder --$_SAVE_FOLDER_PATH
```

Where `$_PATH_TO_DATASET` is the path to the downloaded dataset, and `$_SAVE_FOLDER_PATH` is the path to the directory where the generated files will be saved.

##### __Testing__ - wildplaces

To generate the testing splits run the following command:

```
python generate_splits/testing_sets.py --dataset_root $_PATH_TO_DATASET --save_folder --$_SAVE_FOLDER_PATH
```

This script will generate separate testing pickles for the inter-run and intra-run evaluation modes on each environment.  The **inter-run** pickles will produce query and database files for each testing environment, while the **intra-run** pickles will produce a separate training pickle for each individual point cloud sequence.

##### __Testing__ - anymal, botanic

For ANYmal and botanic, we also use the same way to generate pickle files, which are saved in dataset folders.

#### generate npy files for BEV images

In `scripts/generate_splits/bevwp.py`

generate_bevs function is used to generate BEV images and save .npy files to the folder.

When running the code first time, these files can be generated automatically.

If there is already the .npy file, the dataset load function can skip the generation and directly load it from the file.

?todo how to run it

### model training on Wild-Places

Run the following script to train and evaluate `ForestLPR` network. Specifically, it will train `ForestLPR` network and select a checkpoint that performs best Recall@5 on the validation set as the final model.

use `Deit_multi` as model file

```
python -W ignore train.py --img_size 480 480 --level 1 6 11 --agg gem --depth 12 --inputc 5 --features_dim 1024 --firsteval
```

use `Deit` as model file, train single-BEV version

```
python -W ignore train.py --img_size 480 480 --level 1 6 11 --agg gem --depth 12 --inputc 1 --features_dim 1024 --firsteval
```
### testing
Download the trained checkpoint on Wild-Places from XXX[url](), put it into `$HOME/ForestLPR/runs`, and run the following script to evaluate it.

Wild-Places inter-evaluation

```shell
python -W ignore eval.py --img_size 480 480 --cacheBatchSize 6 --resume ./runs/forestlpr/model_best.pth.tar --level 1 6 11 --dataset inter --subset Karawatha
```

Wild-Places intra-evaluation

```shell
python -W ignore eval.py --img_size 480 480 --cacheBatchSize 12 --resume ./runs/forestlpr/model_best.pth.tar --level 1 6 11 --dataset intra --subset V-03 --world_thresh 3 --time_thresh 600
```

Botanic intra-evaluation

```shell

```

Anymal intra-evaluation

```shell
python -W ignore eval.py --img_size 480 480 --cacheBatchSize 12 --resume ./runs/forestlpr/model_best.pth.tar --level 1 6 11 --dataset intra --subset 3-02 --world_thresh 3 --time_thresh 100
```
### Expected Performance

<table>
  <tr align="center">
    <th rowspan='2'>Type</th><th colspan='3'>V-03</th><th colspan='3'>V-04</th> <th colspan='3'>K-03</th><th colspan='3'>K-04</th>
    </tr>
  <tr align="center">
        <th> R@1 </th> <th> R@5 </th> <th> R@10 </th> <th> R@1 </th> <th> R@5 </th> <th> R@10 </th> <th> R@1 </th> <th> R@5 </th> <th> R@10 </th> <th> R@1 </th> <th> R@5 </th> <th> R@10 </th> 
  </tr>
  <tr align="center">
    <td>V-03</td> <td>8</td> <td>4</td> <td>6</td> <td>1</td> <td>3</td> <td>4</td> <td>8</td> <td>4</td> <td>6</td> <td>1</td> <td>3</td> <td>4</td>
  </tr>
</table>


<table>
  <tr align="center">
    <th rowspan='2'>Type</th><th colspan='3'>Inter-V</th><th colspan='3'>Inter-K</th> <th colspan='3'>Botanic</th><th colspan='3'>ANYmal</th>
    </tr>
  <tr align="center">
        <th> R@1 </th> <th> R@5 </th> <th> R@10 </th> <th> R@1 </th> <th> R@5 </th> <th> R@10 </th> <th> R@1 </th> <th> R@5 </th> <th> R@10 </th> <th> R@1 </th> <th> R@5 </th> <th> R@10 </th> 
  </tr>
  <tr align="center">
    <td>V-03</td> <td>8</td> <td>4</td> <td>6</td> <td>1</td> <td>3</td> <td>4</td> <td>8</td> <td>4</td> <td>6</td> <td>1</td> <td>3</td> <td>4</td>
  </tr>
</table>


## Visulizations
The retrieval results on ANYmal dataset.
<video width="315px" height="150px" src="https://github.com/user-attachments/assets/d6c128c9-4543-4967-bba6-fa707603c238.mp4"></video>



