'''
use PCA in sklearn to do decomposition

Confirm the first-K dim of the dist always has max variance. 
K depends on the 95% representation of the original dist. 


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

    gm_centers = np.random.rand(class_num, feat_dim) # uniform~ (0 ,1)

    center_gather_fisrtN_LSt = [1, 95, 2, 5, 75]
    center_gether_Lst = center_gather_fisrtN_LSt + [0.1] * (feat_dim - len(center_gather_fisrtN_LSt))
    for dim_i in range(feat_dim):
        gm_centers[:, dim_i] *= center_gether_Lst[dim_i]
    
    firstN_stds =  np.array([16.3])
    # np.random.rand(class_num - len(firstN_stds)) * 0.5
    gm_stds = np.random.rand(class_num) * 0.5 + 1 # std for each centers.

    gm_colors = np.random.rand(class_num, 3) * 0.8 + 0.2

    gm_X, gm_y = make_blobs(n_samples = data_amout, n_features = feat_dim, centers = gm_centers,
                  cluster_std = gm_stds , random_state = 9)

    print("[info] data amout: %04d data dim: %04d"%(gm_X.shape[0], gm_X.shape[1]))

    plt.scatter(gm_X[:, 0], gm_X[:, 1], marker='o', s = 10) 
    plt.scatter(gm_centers[:, 0], gm_centers[:, 1], marker='x', s = 25, c = "r") 
    plt.title("Orignal Axis Dist.(First 2 dims)")
    plt.show()


    # gm_pca = PCA(n_components = comp_dim)
    # gm_pca = PCA(n_components = 0.95, svd_solver = "auto")
    gm_pca = PCA(n_components = 20, svd_solver = "auto")
    '''
    auto : auto-selected; 
    randomized: data_num M >> data_dim N; 
    full : basic svd; 
    arpack: sparse
    '''
    gm_pca.fit(gm_X)
    print("auto preserved dims: ", gm_pca.n_components_)
    print("explained ratio:", gm_pca.explained_variance_ratio_)
    print("explained std:", np.sqrt(gm_pca.explained_variance_))

    # X = U * S * V^T; S(21 x 21)

    gm_S = gm_pca.transform(gm_X) # S = U^T * X * V; shape (32560, 2)
    gm_Scenters = gm_pca.transform(gm_centers)

    plt.scatter(gm_S[:, 0], gm_S[:, 1], marker='o',c = "g", s = 10) 
    plt.scatter(gm_Scenters[:, 0], gm_Scenters[:, 1], marker='x', s = 25, c = "r") 
    plt.title("PCA(pure) Axis Dist.(First 2 dims)")
    plt.show()

    gm_X_less = gm_pca.inverse_transform(gm_S)
    plt.scatter(gm_X_less[:, 0], gm_X_less[:, 1], marker='o', s = 10) 
    plt.scatter(gm_centers[:, 0], gm_centers[:, 1], marker='x', s = 25, c = "r") 
    plt.title("Back to Origin Axis Dist.(First 2 dims)") #  X_less = U * S * V^T; shape (32560, 21)
    plt.show()




