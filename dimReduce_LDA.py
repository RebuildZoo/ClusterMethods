'''
LDA (Linear Discriminant Analysis)
use PCA in sklearn to do decomposition

'''
import numpy as np 
np.random.seed(4096)
from sklearn.datasets import make_blobs

# from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

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

    plt.scatter(gm_X[:, 0], gm_X[:, 1], marker='o', s = 10, c = gm_colors[gm_y]) 
    plt.scatter(gm_centers[:, 0], gm_centers[:, 1], marker='x', s = 25, c = "r") 
    plt.title("Orignal Axis Dist with Class Label.(First 2 dims)")
    plt.show()

    fig = plt.figure(figsize=(6,4))
    axes3D = Axes3D(fig)
    axes3D.scatter3D(gm_X[:, 0], gm_X[:, 1], gm_X[:, 2], marker='o',c = gm_colors[gm_y])
    plt.scatter(gm_centers[:, 0], gm_centers[:, 1], gm_centers[:, 2], marker='x',c = 'r')
    plt.title("Orignal Axis Dist with Class Label.(First 3 dims)")
    plt.show()


    gm_lda = LinearDiscriminantAnalysis(n_components = 3)
    gm_lda.fit(gm_X, gm_y)

    gm_S = gm_lda.transform(gm_X)
    gm_Scenters = gm_lda.transform(gm_centers)

    # plt.scatter(gm_S[:, 0], gm_S[:, 1],marker='o',c = gm_colors[gm_y])
    # plt.scatter(gm_Scenters[:, 0], gm_Scenters[:, 1],marker='x',c = 'r')
    # plt.title("LDA Axis Dist.( 2 dims)")
    # plt.show()

    fig = plt.figure(figsize=(6,4))
    axes3D = Axes3D(fig)
    axes3D.scatter3D(gm_S[:, 0], gm_S[:, 1], gm_S[:, 2], marker='o',c = gm_colors[gm_y])
    plt.scatter(gm_Scenters[:, 0], gm_Scenters[:, 1], gm_Scenters[:, 2], marker='x',c = 'r')
    plt.title("LDA Axis Dist.( 3 dims)")
    plt.show()