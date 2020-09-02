'''

'''
import numpy as np 
np.random.seed(4096)
# from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

from sklearn.datasets import make_blobs

import matplotlib.pyplot as plt


if __name__ == "__main__":
    
    class_num = 16
    data_amout = 32560 #4096 * 10; 80 * 407
    feat_dim = 21
    select_feat_dim = 21
    # s_f = 21 : eps = 10; 
    # s_f = 

    data_scale = 50

    gm_centers = np.random.rand(class_num, feat_dim) * data_scale # uniform, (0 ,1)
    gm_stds = np.random.rand(class_num) * 0.5 + 1
    gm_colors = np.random.rand(class_num*3, 3) * 0.8 + 0.2

    gm_X, gm_y = make_blobs(n_samples = data_amout, n_features = feat_dim, centers = gm_centers,
                  cluster_std = gm_stds , random_state = 9)

    gm_X = gm_X[:, :select_feat_dim]
    print("[info] data amout: %04d data dim: %04d"%(gm_X.shape[0], gm_X.shape[1]))

    plt.scatter(gm_X[:, 0], gm_X[:, 1], marker='o', s = 10) 
    plt.scatter(gm_centers[:, 0], gm_centers[:, 1], marker='x', s = 25, c = "r") 
    plt.title("Orignal Blob Dist(First 2 dims). ")
    plt.show()

    gm_cluster = DBSCAN(eps = 9.5, min_samples = 10)
    # gm_cluster.fit(gm_X) # training 

    y_pred = gm_cluster.fit_predict(gm_X) # or gm_cluster.labels_
    n_clusters_pred = len(np.unique(y_pred))
    # center_pred = gm_cluster.cluster_centers_

    plt.scatter(gm_X[:, 0], gm_X[:, 1], c = gm_colors[y_pred.tolist()])
    plt.scatter(gm_centers[:, 0], gm_centers[:, 1], marker='x', s = 25, c = "r") 
    # plt.scatter(center_pred[:, 0], center_pred[:, 1], marker='x', s = 25, c = "k") 
    plt.title("DBSCAN Cluster. use feat_dim = %d, eps = %.1f, #class = %d"%(select_feat_dim, gm_cluster.eps, n_clusters_pred))
    plt.show()
