
# ForestLPR: LiDAR Place Recognition in Forests Attentioning Multiple BEV Density Images

The code is almost there. Don't ask, we don't know either. Good luck!😉 [kidding]

### TODO

- arxiv version upload and insert the link
- toc

source code of our CVPR'25 paper [ForestLPR: LiDAR Place Recognition in Forests Attentioning Multiple BEV Density Images]()

<img src="https://github.com/user-attachments/assets/21c49ddd-6f88-4237-bfe9-ef78bcd0e39c" width="400px">

## Table of Contents

[toc]

## Environments

- CUDA 11.8
- Python 3.9.4
- PyTorch 2.0.1

We used Anaconda to setup a deep learning workspace that supports PyTorch. Run the following script to install the required packages.

```shell
conda env create -f environment.yml
conda activate forestlpr
git clone git@github.com:shenyanqing1105/ForestLPR-CVPR2025.git
cd ForestLPR-CVPR2025
pip install -r requirements.txt
conda deactivate
```

## Required Submap Data

We use two public datasets and one own-collected dataset: [Wild-Places](https://github.com/csiro-robotics/Wild-Places), [Botanic Garden Dataset](https://github.com/robot-pesg/BotanicGarden), and ANYmal dataset. Additionally, in our work, we have processed the datasets into cropped point clouds `Cloud_normalized`, `pickle` files and BEV density images `.npy` files, which is more convenient to test the BEV-based methods.

For [Botanic](https://www.123865.com/s/tp6jTd-jGFbh?提取码:MPTw), and [ANYmal](https://www.123865.com/s/tp6jTd-6GFbh?提取码:6hXL), we repurpose the dataset and use GT to register multiple frames to 1 submap.

For [Wild-Places](https://www.123865.com/s/tp6jTd-TPFbh?提取码:s0u3), we provide the additional processed files, and the data should be merged based on the sequences.

Environment visualization
![environments](https://github.com/user-attachments/assets/e799831c-b655-49e8-b4dd-0c4ca00539dc)

## dataset structure

```
$_PATH_TO_DATASET/
├── anymal/		  # Dataset
|   ├── anymal-3/ # Environment
|	│   └── 02/   # Sequence
|	|	    ├── Clouds_downsampled/
|	|	    ├── Clouds_normalized/
|	|	    └── poses_aligned.csv
|   └── testing/
|	    ├── 3-02.pickle
|	    └── .npy files
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
|	    └── .npy files
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
    |	└── .npy files
    ├── training/
    |	├── training_wild-places.pickle
    |	├── testing_wild-places.pickle
    |	└── .npy files
    └── Venman/
            ├── V-01/
            ├── V-02/
            ├── V-03/
            └── V-04/
```

## File structure

```
$HOME/ForestLPR
├── dataset/		  # Dataset
|   ├── anymal-3/ # Environment
|	│   └── 02/   # Sequence
|	|	    ├── Clouds_downsampled/
|	|	    ├── Clouds_normalized/
|	|	    └── poses_aligned.csv
|   └── testing/
|	    ├── 3-02.pickle
|	    └── npy files
├── models/
|   ├── model/    # model files of ForestLPR
|	|	├── Deit.py
|	|	├── Deit_multi.py
|	|	└── ...
|   └── bp/    # model files of BevPlace
├── pickle_files/		  
├── runs/		  # checkpoints
├── scripts/
    ├── eval/    # evaluations for 2D baselines
    |	├── mapclosure/
    |   |	├── mc_inter_sequence.py
    |   |	├── mc_intra_sequence.py
    |   |	├── mc_utils.py
    |   |	├── mc_eval_config_anymal.yaml
    |   |	├── mc_eval_config_botanic.yaml
    |	|	└── mc_eval_config_wp.yaml
    |	└── scan_context/
    |   |	├── sc_inter_sequence.py
    |   |	├── sc_intra_sequence.py
    |   |	├── sc_utils.py
    |   |	├── sc_eval_config_anymal.yaml
    |   |	├── sc_eval_config_botanic.yaml
    |		└── sc_eval_config_wp.yaml
    |	generate_splits/
    |   ├── bevwp.py    # dataloader of ForestLPR
    |   ├── ground_filter.py    # point-cloud preprocessing: ground segmentation and height offset removal
    |   ├── merge.py    # generating the submaps from original LiDAR scans
    |   ├── testing_sets.py
    |   ├── training_sets.py
    |   ├── util.py
    |	└── utils.py
    ├── load_pointcloud.py
    └── overlap.py    # compute the overlap between two scans using octree
├── eval.py    # evaluations for forestLPR
├── eval_bp.py    # evaluations for BevPLace
├── train.py
├── util.py
└── utils.py

$HOME/LoGG3D-Net
├── [original code]
├── scripts/
|   ├── inter_sequence.py
|   ├── intra_sequence.py
|   ├── load_pointcloud.py
|   ├── transloc_utils.py
|	└── util.py
└── checkpoints

$HOME/MinkLoc3Dv2
├── [original code]
├── scripts
|   ├── inter_sequence.py
|   ├── intra_sequence.py
|   ├── load_pointcloud.py
|   ├── minkloc_utils.py
|	└── util.py
└── checkpoints

$HOME/TransLoc3D
├── [original code]
├── scripts
|   ├── inter_sequence.py
|   ├── intra_sequence.py
|   ├── load_pointcloud.py
|   ├── logg3d_utils.py
|	└── utils.py
└── checkpoints

```

## Scripts

First check dataset file path in bevwp.py

### Pre-processing Point Clouds

Run `scripts/generate_splits/ground_filter.py`, change `<ROOT_PATH>` and `<OUT_PATH>`

Including ground segmentation and height offeset removal

### Generating Pickle Files for Each Sequence

#### __Training__ - wildplaces

To generate the training splits run the following command:

```
python generate_splits/training_sets.py --dataset_root $_PATH_TO_DATASET --save_folder --$_SAVE_FOLDER_PATH
```

Where `$_PATH_TO_DATASET` is the path to the downloaded dataset, and `$_SAVE_FOLDER_PATH` is the path to the directory where the generated files will be saved.

#### __Testing__ - wildplaces

To generate the testing splits run the following command:

```
python generate_splits/testing_sets.py --dataset_root $_PATH_TO_DATASET --save_folder --$_SAVE_FOLDER_PATH
```

This script will generate separate testing pickles for the inter-run and intra-run evaluation modes on each environment.  The **inter-run** pickles will produce query and database files for each testing environment, while the **intra-run** pickles will produce a separate training pickle for each individual point cloud sequence.

##### __Testing__ - anymal, botanic

For ANYmal and botanic, we also use the same way to generate pickle files, which are saved in dataset folders.

#### Generate BEV Images and .npy files

In `scripts/generate_splits/bevwp.py`, set `TOP_Z_MAX` and `TOP_Z_MIN` to crop the point cloud. `TOP_Z_DIVISION` refers to height interval of slices.

`generate_bevs function` is used to generate BEV images and save .npy files to the folder.

This script is also used for dataloader. When running the code first time, these files can be generated automatically. If there is already the .npy file, the dataset load function can skip the generation and directly load it from the file.

## Model Training on Wild-Places

Run the following script to train and evaluate `ForestLPR` network. Specifically, it will train `ForestLPR` network and select a checkpoint that performs best Recall@5 on the validation set as the final model.

use `Deit_multi` as model file

```
python -W ignore train.py --img_size 480 480 --level 1 6 11 --agg gem --depth 12 --inputc 5 --features_dim 1024 --firsteval
```

use `Deit` as model file, train single-BEV version

```
python -W ignore train.py --img_size 480 480 --level 1 6 11 --agg gem --depth 12 --inputc 1 --features_dim 1024 --firsteval
```

## Model Testing

Download the trained model on Wild-Places from [checkpoints](https://www.123865.com/s/tp6jTd-yGFbh?提取码:ZM3W), put it into `$HOME/ForestLPR/runs/forestlpr`, and run the following script to evaluate it.

Change dataset file path in bevwp.py

**Wild-Places inter-evaluation**

```shell
python -W ignore eval.py --img_size 480 480 --cacheBatchSize 6 --resume ./runs/forestlpr/checkpoints/model_best.pth.tar --level 1 6 11 --dataset inter --subset Karawatha
```

**Wild-Places intra-evaluation**

```shell
python -W ignore eval.py --img_size 480 480 --cacheBatchSize 12 --resume ./runs/forestlpr/checkpoints/model_best.pth.tar --level 1 6 11 --dataset intra --subset V-03 --world_thresh 3 --time_thresh 600
```

**Botanic intra-evaluation**

```shell
python -W ignore eval.py --img_size 480 480 --cacheBatchSize 12 --resume ./runs/forestlpr/checkpoints/model_best.pth.tar --level 1 6 11 --dataset intra --subset 5-03 --world_thresh 3 --time_thresh 100
```

**Anymal intra-evaluation**

```shell
python -W ignore eval.py --img_size 480 480 --cacheBatchSize 12 --resume ./runs/forestlpr/checkpoints/model_best.pth.tar --level 1 6 11 --dataset intra --subset 3-02 --world_thresh 3 --time_thresh 100
```

## Baselines Testing

Here we provide the Code of Baselines for testing Wild-Places, helping others reproduce the results to compare.

[Transloc3D](https://www.dropbox.com/s/dotyuurc3n5m24w/TransLoc3D.pth?dl=0)

[LOGG3D-Net](https://www.dropbox.com/s/h1ic00tvfnstvfm/LoGG3D-Net.pth?dl=0)

[MinkLOc3Dv2](https://www.dropbox.com/s/8ijq9h99m1snzxn/MinkLoc3Dv2.pth?dl=0)

[BevPlace](https://drive.google.com/file/d/1AhCAhI70CVfwioMSOumiCFlUuF9-0WYP/view?usp=sharing)

### 3D Methods

Download the original code (or use git submodule) and the added  xx_ `script `folder. The key files are `inter-sequence.py `and `intra-sequence.py `. Put the pretrained model into the `checkpoints` folder. Move xx_ `script `folder. (e.g., move `transloc_scripts` into TransLoc3D, and rename it to `scripts`)

**TransLoc3D** on Wild-Places, intra

Note: the `cfg.model_cfg.quantization_size` has been changed compared with reproduction in Wild-Places, for better performance.

```
cd scriptsig ../configs/transloc3d_baseline_cfg.py --databases $_PATH_TO_DATASET/Wild-Places/testing/K-04.pickle --run_names Karawatha/K-04 --save_dir ../dataset/Wild-Places/testing/ --database_featu
python intra-sequence.py --confres None --root $_PATH_TO_DATASET/Wild-Places --quan_size 1
```

`quan_size` is used to set `cfg.model_cfg.quantization_size`

**Logg3D-Net** on Botanic, intra

```
cd scripts
python intra-sequence.py --databases $_PATH_TO_DATASET/botanic/testing/5-03.pickle --run_names 1005-03/02 --save_dir ../dataset/botanic/testing/ --database_features None --root $_PATH_TO_DATASET/botanic --time_thresh 100 --world_thresh 3
```

**Logg3D-Net** on Wild-Places, inter

```
cd scripts
python inter-sequence.py --databases $_PATH_TO_DATASET/Wild-Places/testing/Venman_evaluation_database.pickle --queries $_PATH_TO_DATASET/Wild-Places/testing/Venman_evaluation_query.pickle --location_names Venman --query_features None --database_features None --root $_PATH_TO_DATASET/Wild-Places
```

**MinkLoc3Dv2** on ANYmal, intra

use `v2` configuration

```
cd scripts
python intra-sequence.py --databases $_PATH_TO_DATASET/anymal/testing/3-02.pickle --run_names anymal/03-2 --save_dir ../dataset/Wild-Places/testing/ --database_fe
atures None --root $_PATH_TO_DATASET/anymal --config ../config/config_baseline_v2.txt --model_config ../models/minkloc3dv2.txt --weights ../checkpoints/MinkLoc3Dv2.pth --world_thresh 3 --time_thresh 100
```

### 2D Methods

**Scan-Context**

```
cd scripts/eval/scan_context & python sc_intra_sequence.py --config ./sc_eval_config_anymal.yaml
```

**MapClosure**

```
python mc_intra_sequence.py --config mc_eval_config_wp.yaml
```

**BevPlace**

```
python eval_bp.py --dataset intra --cacheBatchSize 2 --img_size 480 480 --features_dim 8192 --subset K-03 --resume ./runs/bevplace/checkpoint_wp.tar --world_thresh 1
```

## Expected Performance

<table>
  <tr align="center">
    <th rowspan='2'>Type</th><th colspan='2'>V-03</th><th colspan='2'>V-04</th> <th colspan='2'>K-03</th><th colspan='3'>K-04</th> <th colspan='2'>ANYmal</th>
    </tr>
  <tr align="center">
        <th> R@1 </th> <th> F1 </th> <th> R@1 </th> <th> F1 </th> <th> R@1 </th> <th> F1 </th>  <th> R@1 </th> <th> F1</th> <th> R@1 </th> <th> F1</th>
  </tr>
  <tr align="center">
    <td>ForestLPR</td> <td>76.53</td> <td>64.15</td> <td>82.33</td> <td>78.62</td> <td>74.89</td> <td>65.01</td> <td>76.73</td> <td>81.97</td> <td>71.87</td> <td>81.45</td> 
  </tr>
</table>

<table>
  <tr align="center">
    <th rowspan='2'>Type</th><th colspan='1'>Inter-V</th><th colspan='1'>Inter-K</th> <th colspan='1'>Validation</th> <th colspan='2'>Botanic05-3</th> <th colspan='2'>Botanic05-6</th>
    </tr>
  <tr align="center">
        <th> R@1 </th> <th> R@1 </th> <th> R@1 </th> <th> R@1 </th> <th> F1 </th> <th> R@1 </th> <th> F1 </th> 
  </tr>
  <tr align="center">
   <td>ForestLPR</td> <td>77.14</td> <td>79.02</td> <td>80.03</td> <td>84.81</td> <td>78.21</td> <td>82.00</td> <td>78.82</td>
  </tr>
</table>

## Visulizations

The retrieval video on ANYmal dataset.

<video width="315px" height="150px" src="https://github.com/user-attachments/assets/d6c128c9-4543-4967-bba6-fa707603c238.mp4"></video>

## Reference

If you find the repo useful, please consider citing our paper:

```
@inproceedings{forestlpr,
  title={ForestLPR: LiDAR Place Recognition in Forests Attentioning Multiple BEV Density Images},
  author={Shen, Yanqing and Tuna, Turcan and Cadena, Cesar and Hutter, Marco and Zheng, Nanning},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
  year={2025}
}
```

## Acknowledgments

We would like to sincerely thank [Wild-Places](https://github.com/csiro-robotics/Wild-Places) for the awesome released code.
