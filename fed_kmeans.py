import numpy as np
from sklearn.cluster import KMeans,MiniBatchKMeans
from sklearn.datasets import load_iris,load_sample_image
from scipy.optimize import linear_sum_assignment

from matplotlib import pyplot as plt
from matplotlib.image import imread

def avgcentroids(centroids):
    '''
    input:
    centroids: list of numpy arrays (n_clients, n_clusters, n_features)
    output:
    avg_centroids: numpy array (n_clusters, n_features)
    '''
    n_clients = centroids.shape[0]
    n_clusters = centroids.shape[1]
    init_centroids = centroids[0]
    avg_centroids = init_centroids.copy()
    for i in range(1,n_clients):
        new_centroids = centroids[i]
        # Calculate pairwise distances between init_centroids and new_centroids
        distances = np.linalg.norm(init_centroids[:, np.newaxis, :] - new_centroids[np.newaxis, :, :], axis=2)
        
        # Find the optimal alignment using the Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(distances)
        
        # Reorder new_centroids to align with init_centroids
        new_centroids = new_centroids[col_ind]
        avg_centroids = avg_centroids + new_centroids
    avg_centroids = avg_centroids / n_clients
    return avg_centroids

def fed_kmeans_exp_k(data=None, n_clients=2, n_clusters=3, random_state=37):
    '''
    experiment on impact of clients_num on the performance of federated kmeans
    '''
    if data is None:
        dataset = load_iris()
        data = dataset['data']

    n_nodes = data.shape[0]
    input_dim = data.shape[1]
    random_index = np.random.permutation(n_nodes)
    shuffled_data = data[random_index]
    datas = []
    kmeans_list = []
    centroids = []
    for i in range(n_clients):
        # Split data into n_clients parts
        # Each client gets n_nodes // n_clients samples 
        start = i * n_nodes // n_clients
        end = (i + 1) * n_nodes // n_clients
        datas.append(shuffled_data[start:end])
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state).fit(datas[i])
        kmeans_list.append(kmeans)
        centroids.append(kmeans.cluster_centers_)
        del kmeans
    centroids = np.array(centroids)
    #平均聚类中心
    avg_centroids = avgcentroids(centroids)
    #全局聚类
    kmeans_global = KMeans(n_clusters=n_clusters, random_state=37).fit(data)
    global_centroids = kmeans_global.cluster_centers_
    distances = np.linalg.norm(global_centroids[:, np.newaxis, :] - avg_centroids[np.newaxis, :, :], axis=2)
    #对齐平均聚类中心和全局聚类中心，因为kmeans的聚类中心是随机顺序的，全局聚类中心和局部聚类中心之间按照距离最小来匹配
    row_ind, col_ind = linear_sum_assignment(distances)
    avg_centroids = avg_centroids[col_ind]
    avg_kmeans = KMeans(n_clusters=n_clusters, random_state=random_state).fit(data)
    avg_kmeans.cluster_centers_ = avg_centroids
    label = avg_kmeans.predict(data)
    mse = np.mean(np.linalg.norm(global_centroids - avg_centroids, axis=1))
    nmse = np.mean(np.linalg.norm(global_centroids - avg_centroids, axis=1) / np.linalg.norm(global_centroids, axis=1))
    acc = (label == kmeans_global.labels_).sum() / len(label)
    picture_avg = label.reshape(633,952)
    picture_globle = kmeans_global.labels_.reshape(633,952)

    print("acc:", (label== kmeans_global.labels_).sum()/len(label),"accs:", (label == kmeans_global.labels_).sum())

    plt.subplot(1, 3, 1)
    plt.imshow(data.reshape(633,952,3))
    plt.title('original picture')
    plt.subplot(1, 3, 2)
    plt.imshow(picture_avg)
    plt.title('Avg KMeans')
    plt.subplot(1, 3, 3)
    plt.imshow(picture_globle)
    plt.legend()
    plt.xlabel(f"MSE: {mse:.2e}, NMSE: {nmse:.2e}")
    plt.show()
    return mse, nmse, acc
if __name__ == "main":
    avg_mse, avg_nmse, avg_acc = 0, 0, 0 
    n_clients = 5
    n_clusters = 7
    data = imread('picture4_28.png')[:,:,:3].reshape(-1,3)
    for i in range(10):
        mse,nmse,acc = fed_kmeans_exp_k(data,n_clients=n_clients, n_clusters=n_clusters, random_state=i)
        avg_mse += mse
        avg_nmse += nmse
        avg_acc += acc
    avg_mse /= 10
    avg_nmse /= 10
    avg_acc /= 10

    print(f"Average MSE: {avg_mse:.4f}")
    print(f"Average NMSE: {avg_nmse:.4f}")
    print(f"Average accuracy: {avg_acc:.4f}" )