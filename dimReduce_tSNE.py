'''
t-SNE(t-distributed Stochastic neighbour Embedding)

For the buit-in t-SNE in sklearn is too slow , 
we use https://github.com/KlugerLab/FIt-SNE

'''

import numpy as np 
np.random.seed(4096)
# from sklearn.datasets import make_blobs

import sys 
sys.path.append("FIt-SNE/")
# r"D:\Documents\Git_Repo\Fork\utils\FIt-SNE"
import os
import subprocess
import struct
import numpy as np
from fast_tsne import fast_tsne
import matplotlib.pyplot as plt
plt.style.use('ggplot')


PLTCOLOR_ARR = np.array(['#a6cee3', # light blue
                    '#1f78b4', # blue
                    '#b2df8a', # lighrt green
                    '#33a02c', # green 
                    '#fb9a99', # light red
                    '#e31a1c', # red
                    '#fdbf6f', # light orange
                    '#ff7f00', # orange
                    '#cab2d6', # light purple
                    '#6a3d9a', # purple
                    '#f59adb', # light pink 
                    '#db1da5', # pink 
                    '#9af5ed', # light cyan
                    '#09e8d4', # cyan
                    ])

def operate_tsne(data_Lst):

    len_Lst = list(map(len, data_Lst))
    res_Arr = fast_tsne(np.vstack(data_Lst)) # (N, 2)
    plot_scatters(res_Arr, len_Lst)

def plot_scatters(data2D, class_num_Lst):
    '''
    data2D (N, 2)
    class_num_Lst = [a , b, c, d, ...]
    first class is the lagerst. 
    '''
    assert len(data2D) == sum(class_num_Lst)
    plt.figure(figsize=(7,7))
    
    index_begin = 0
    for class_i, index_interval_i in enumerate(class_num_Lst):
        scale = 5
        if class_i == 0: scale = 10
        plt.scatter(data2D[index_begin: (index_begin + index_interval_i), 0], data2D[index_begin: (index_begin + index_interval_i):, 1], marker='o', s = 10, c = PLTCOLOR_ARR[class_i]) 

        index_begin += index_interval_i

    plt.title("tSNE 2D Axis Dist.")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    sample_num = int(5e4)
    feat_dim = 50

    gm_X1 = np.random.uniform(0, 2, (sample_num, feat_dim))
    gm_X2 = np.random.uniform(-1, 1, (sample_num, feat_dim))
    gm_X3 = np.random.rand(sample_num, feat_dim)
    gm_X4 = np.random.rand(sample_num, feat_dim) * 3

    operate_tsne([gm_X1, gm_X2, gm_X3, gm_X4])
    