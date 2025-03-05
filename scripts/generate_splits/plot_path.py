# coding=utf-8
'''
Author: shenyanqing1105 1159364090@qq.com
Date: 2025-03-05 11:34:46
LastEditors: shenyanqing1105 1159364090@qq.com
LastEditTime: 2025-03-05 16:24:24
FilePath: /ForestLPR-CVPR2025/scripts/generate_splits/plot_path.py
'''
import pandas as pd
from os.path import join
import matplotlib.pyplot as plt

root_path = '/dataset/anymal/01/'
dataset = 'anymal'
df = pd.read_csv(join(root_path,'poses_aligned.csv'), delimiter = ',', dtype = str)

# load pose_aligned.csv


x_values = df['x'].values.tolist()
x_values = [float(item) for item in x_values]
y_values = df['y'].values.tolist()
y_values = [float(item) for item in y_values]

plt.plot(x_values, y_values, marker='o', linestyle='-', color='b')
plt.savefig(join('../plot', "path_{}.png".format(dataset)))