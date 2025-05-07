import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
def avgcentroids(centroids):
    '''
    input:
    centroids: list of numpy arrays (n_clients, n_clusters, n_features)
    output:
    avg_centroids: numpy array (n_clusters, n_features)
    '''
    n_clients = len(centroids)
    n_clusters = centroids[0].shape[0]
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

class Server:
    def __init__(self, client_list):
        self.client_list = client_list
        params = []
        client = self.client_list[len(self.client_list)-1]
        
        params += params + [{'params': encoder.parameters()} for encoder in client.encoders] + [{'params': decoder.parameters()} for decoder in client.decoders] + [{'params': client.Z_fc.parameters()}]
        self.optimizer = torch.optim.Adam(
            params,
            lr=0.001,
            weight_decay=1e-4
        )
        
        
    def fed_kmeans(self):
        centroids = []
        for client in self.client_list:
            centroids.append(client.centroids)
        fed_kmeans_centroid = avgcentroids(centroids)
        return fed_kmeans_centroid
    
    def avg_loss(self):
        total_loss = 0
        for client in self.client_list:
            total_loss += client.loss
        average_loss = total_loss / len(self.client_list)
        return average_loss
