'''
use K-means in Sklearn to do cluster. 

data amount : ~ Freihand Datasets; 
#images = 32560
#bones = 21

'''
import numpy as np 
np.random.seed(4096)
from sklearn.cluster import KMeans

from sklearn.datasets import make_blobs

import matplotlib.pyplot as plt



if __name__ == "__main__":
    
    class_num = 16
    data_amout = 32560 #4096 * 10; 80 * 407
    feat_dim = 21

    data_scale = 50

    gm_centers = np.random.rand(class_num, feat_dim) * data_scale # uniform, (0 ,1)
    gm_stds = np.random.rand(class_num) * 0.5 + 0.7 
    gm_colors = np.random.rand(class_num, 3) * 0.8 + 0.2

    gm_X, gm_y = make_blobs(n_samples = data_amout, n_features = feat_dim, centers = gm_centers,
                  cluster_std = gm_stds , random_state = 9)

    print("[info] data amout: %04d data dim: %04d"%(gm_X.shape[0], gm_X.shape[1]))

    plt.scatter(gm_X[:, 0], gm_X[:, 1], marker='o', s = 10) 
    plt.scatter(gm_centers[:, 0], gm_centers[:, 1], marker='x', s = 25, c = "r") 

    plt.show()


    # random_state : random seed ; max_iter = if the center don't converge, break at ... 
    gm_cluster = KMeans(n_clusters = class_num , random_state = 9, max_iter = 300) 

    gm_cluster.fit(gm_X) # training 

    y_pred = gm_cluster.predict(gm_X)
    center_pred = gm_cluster.cluster_centers_
    
    plt.scatter(gm_X[:, 0], gm_X[:, 1], c = gm_colors[y_pred.tolist()])
    plt.scatter(gm_centers[:, 0], gm_centers[:, 1], marker='x', s = 25, c = "r") 
    plt.scatter(center_pred[:, 0], center_pred[:, 1], marker='x', s = 25, c = "k") 
    plt.show()


