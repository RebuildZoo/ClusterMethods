'''
use PCA in sklearn to do decomposition

'''
import numpy as np 
np.random.seed(4096)
from sklearn.datasets import make_blobs

from sklearn.decomposition import PCA

import matplotlib.pyplot as plt


if __name__ == "__main__":

    class_num = 16
    data_amout = 32560 #4096 * 10; 80 * 407
    feat_dim = 21
    comp_dim = 3
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


    # gm_pca = PCA(n_components = comp_dim)
    gm_pca = PCA(n_components = 0.95)
    gm_pca.fit(gm_X)
    print("auto selected N: ", gm_pca.n_components_)
    print(gm_pca.explained_variance_ratio_)
    print(gm_pca.explained_variance_)

    # X = U * S * V; S(21 x 21)

    gm_S = gm_pca.transform(gm_X) # S = 
    gm_X_less = gm_pca.inverse_transform(gm_S)






