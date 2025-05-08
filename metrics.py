import torch
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
CUDA_LAUNCH_BLOCKING=1
def modularity(labels, edge_index, ):
    """
    计算图的Modularity
    :param labels: 节点聚类结果标签
    :param edge_index: 边索引，将转化为邻接矩阵
    :return: Modularity值
    """
    # 构建邻接矩阵
    num_nodes = labels.shape[0]
    adj_matrix = torch.zeros((num_nodes, num_nodes))  # 初始化邻接矩阵
    adj_matrix[edge_index[0], edge_index[1]] = 1  # 填充边
    m = edge_index.size(1) / 2  # 边的数量

    A = adj_matrix  # 邻接矩阵
    adj_matrix = adj_matrix.numpy()
    k = A.sum(dim=1)  # 节点的度
# 初始化 Modularity 值
    Q = 0
    # 首先创建一个布尔矩阵，用于标记节点对是否在同一簇
    same_cluster = (labels.unsqueeze(1) == labels.unsqueeze(0))
    # 计算度矩阵的外积
    k_outer = torch.outer(k, k)
    # 计算分子部分
    numerator = A - k_outer / (2 * m)
    numerator = numerator.to(same_cluster.device)
    # 筛选出同一簇的元素并求和
    Q = (numerator * same_cluster).sum() / (2 * m)
    return Q.item()  # 返回Modularity值

def density(labels, edge_index):
    """
    计算图的Density
    :param labels: 节点聚类结果标签
    :param edge_index: 边索引，将转化为邻接矩阵
    :return: Density值
    """
    L = edge_index.size(1)   # 边的数量
    # 提取边的两个端点的标签
    src_labels = labels[edge_index[0]]
    dst_labels = labels[edge_index[1]]

    # 检查边的两个端点是否属于同一标签
    same_label_edges = src_labels == dst_labels

    # 计算同一标签的边的数量
    same_label_edge_count = same_label_edges.sum().item()
    return same_label_edge_count / L


def feature_similarity(labels, edge_index, features):
    """
    计算所有相同label下的节点的特征相似度的平均值
    :param labels: 节点聚类结果标签
    :param edge_index: 边索引，将转化为邻接矩阵
    :param features: 节点的特征矩阵
    :return: 所有标签下节点特征的平均相似度
    """
    unique_labels = torch.unique(labels)
    total_similarity = 0
    valid_label_count = 0

    for label in unique_labels:
        # 筛选出相同标签的节点索引
        label_indices = torch.where(labels == label)[0]
        # 提取相同标签节点的特征
        label_features = features[label_indices].cpu().numpy()

        if len(label_features) > 1:
            # 计算余弦相似度矩阵
            sim_matrix = cosine_similarity(label_features)
            # 排除对角线元素（自身与自身的相似度）
            non_diag_indices = ~np.eye(len(sim_matrix), dtype=bool)
            # 计算平均相似度
            avg_similarity = sim_matrix[non_diag_indices].mean()
            total_similarity += avg_similarity
            valid_label_count += 1
        elif len(label_features) == 1:
            # 只有一个节点时，相似度设为 1
            total_similarity += 1.0
            valid_label_count += 1

    if valid_label_count == 0:
        return 0.0
    else:
        return total_similarity / valid_label_count
