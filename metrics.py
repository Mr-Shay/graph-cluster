import torch
import numpy
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

