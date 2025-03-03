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

## Required Data

We use two public datasets and one own-collected dataset: [Wild-Places](https://github.com/csiro-robotics/Wild-Places), [Botanic Garden Dataset](https://github.com/robot-pesg/BotanicGarden), and [ANYmal](). The extracted feature is placed  in `$HOME/xxxx/`.

Additionally, in our work, we have processed the datasets into cropped point clouds [download]() and BEV density images [download](), which is more convenient to test the BEV-based methods.

todo environment visualization here

## Codebase

Here we provide the Code of Baselines for testing Wild-Places. todo align

## File structure


## Scripts

### Pre-processing point clouds
which file, how to run, how to set the hpyer-parameters

### model training on Wild-Places
Run the following script to train and evaluate `ForestLPR` network. Specifically, it will train `ForestLPR` network and select a checkpoint that performs best Recall@5 on the validation set as the final model.

### testing
Download trained checkpoint on Wild-Places from XXX[url]() and run the following script to evaluate it.

Wild-Places inter-evaluation

```shell

```

Wild-Places intra-evaluation

```shell

```

Botanic intra-evaluation

```shell

```

Anymal intra-evaluation

```shell

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



