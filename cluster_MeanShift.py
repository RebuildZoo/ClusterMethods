'''

'''

import numpy as np 
np.random.seed(4096)
from sklearn.cluster import MeanShift, estimate_bandwidth

from sklearn.datasets import make_blobs

import matplotlib.pyplot as plt


if __name__ == "__main__":

    class_num = 16
    data_amout = 32560 #4096 * 10; 80 * 407
    feat_dim = 21
    select_feat_dim = 8
    
    data_scale = 50

    gm_centers = np.random.rand(class_num, feat_dim) * data_scale # uniform, (0 ,1)
    gm_stds = np.random.rand(class_num) * 0.5 + 1
    gm_colors = np.random.rand(class_num*10, 3) * 0.8 + 0.2

    gm_X, gm_y = make_blobs(n_samples = data_amout, n_features = feat_dim, centers = gm_centers,
                  cluster_std = gm_stds , random_state = 9)
    
    gm_X = gm_X[:, :select_feat_dim]
    print("[info] data amout: %04d data dim: %04d"%(gm_X.shape[0], gm_X.shape[1]))

    plt.scatter(gm_X[:, 0], gm_X[:, 1], marker='o', s = 10) 
    plt.scatter(gm_centers[:, 0], gm_centers[:, 1], marker='x', s = 25, c = "r") 
    plt.title("Orignal Blob Dist(First 2 dims). ")
    plt.show()

    bandwidth_estm = estimate_bandwidth(gm_X,  quantile = 0.2, n_samples = 1000, random_state = 0, n_jobs = 2) # scalar
    # band-width represent the 'cluster resolution of the process; smaller bw always leads to more cluster class number. '
    gm_cluster = MeanShift(bandwidth = 6.729  , bin_seeding = True) # , bin_seeding = True, n_jobs = 2; 
    gm_cluster.fit(gm_X)
    center_pred = gm_cluster.cluster_centers_
    y_pred = gm_cluster.predict(gm_X)
    n_clusters_pred = len(np.unique(y_pred))
    print("number of estimated clusters: %d, bw: %.3f, est_bw: %.3f"%(n_clusters_pred, gm_cluster.bandwidth, bandwidth_estm))

    plt.scatter(gm_X[:, 0], gm_X[:, 1], marker='o', s = 10, c = gm_colors[y_pred.tolist()]) 
    plt.scatter(gm_centers[:, 0], gm_centers[:, 1], marker='x', s = 25, c = "r") 
    plt.scatter(center_pred[:, 0], center_pred[:, 1], marker='x', s = 25, c = "k") 
    plt.title("MeanShift Cluster. feat_dim = %d, BW = %.3f, #class = %d"%(select_feat_dim, gm_cluster.bandwidth, n_clusters_pred))
    plt.show()

