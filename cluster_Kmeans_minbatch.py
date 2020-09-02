'''
use mini-batch K-means in Sklearn to do cluster. 
#images = 32560

'''
import numpy as np 
np.random.seed(4096)
from sklearn.datasets import make_blobs

from sklearn.cluster import MiniBatchKMeans

import matplotlib.pyplot as plt


if __name__ == "__main__":

    class_num = 16
    data_amout = 32560 #4096 * 10; 80 * 407
    BS = 407
    n_batch = data_amout // BS

    feat_dim = 21

    data_scale = 50

    gm_centers = np.random.rand(class_num, feat_dim) * data_scale # uniform, (-1 ,1)
    gm_stds = np.random.rand(class_num) * 0.5 + 1
    gm_colors = np.random.rand(class_num, 3) * 0.8 + 0.2

    gm_X_flat, gm_y_flat = make_blobs(n_samples = data_amout, n_features = feat_dim, centers = gm_centers,
                  cluster_std = gm_stds , random_state = 9)

    gm_X = gm_X_flat.reshape((n_batch, BS, feat_dim))
    gm_y = gm_y_flat.reshape((n_batch, BS))

    plt.scatter(gm_X_flat[:, 0], gm_X_flat[:, 1], marker='o', s = 10) 
    plt.scatter(gm_centers[:, 0], gm_centers[:, 1], marker='x', s = 25) 
    plt.title("Orignal Dist. ")
    plt.show()

    # perform MiniBatchKMeans
    gm_cluster = MiniBatchKMeans(n_clusters = class_num, random_state = 9, batch_size= BS)
    for batch_idx in range(len(gm_X)):
        gm_cluster.partial_fit(gm_X[batch_idx])
    
    y_pred_flat = gm_cluster.predict(gm_X_flat)
    center_pred = gm_cluster.cluster_centers_

    plt.scatter(gm_X_flat[:, 0], gm_X_flat[:, 1], c = gm_colors[y_pred_flat.tolist()], s = 10)

    plt.scatter(gm_centers[:, 0], gm_centers[:, 1], marker='x', s = 25, c = "r") 
    plt.scatter(center_pred[:, 0], center_pred[:, 1], marker='x', s = 25, c = "k") 
    plt.title("K-Means(batch) Cluster Res. ")
    plt.show()
