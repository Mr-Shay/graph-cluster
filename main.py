import argparse
import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
import torch_geometric.transforms as T
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
# custom modules
from MaskGAE import MaskGAE, DegreeDecoder, EdgeDecoder, GNNEncoder
from mask import MaskEdge
from torch_geometric.datasets import Planetoid
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from matplotlib import pyplot as plt
from client import Client
from server import Server
from metrics import *
def set_seed(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

def train_nodeclas(model, data, args, device='cpu'):
    @torch.no_grad()
    def test(loader):
        clf.eval()
        logits = []
        labels = []
        for nodes in loader:
            logits.append(clf(embedding[nodes]))
            labels.append(y[nodes])
        logits = torch.cat(logits, dim=0).cpu()
        labels = torch.cat(labels, dim=0).cpu()
        logits = logits.argmax(1)
        return (logits == labels).float().mean().item()

    if args.dataset in {'arxiv', 'products', 'mag'}:
        batch_size = 4096
    else:
        batch_size = 512
        
    train_loader = DataLoader(data.train_mask.nonzero().squeeze(), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(data.test_mask.nonzero().squeeze(), batch_size=16, drop_last=True)
    val_loader = DataLoader(data.val_mask.nonzero().squeeze(), batch_size=16, drop_last=True)

    data = data.to(device)
    y = data.y.squeeze()
    embedding = model.encoder.get_embedding(data.x, data.edge_index)

    if args.l2_normalize:
        embedding = F.normalize(embedding, p=2, dim=1)  # Cora, Citeseer, Pubmed    

    loss_fn = nn.CrossEntropyLoss()
    clf = nn.Linear(embedding.size(1), y.max().item() + 1).to(device)

    results = []
    
    for run in range(1, args.runs+1):
        nn.init.xavier_uniform_(clf.weight.data)
        nn.init.zeros_(clf.bias.data)
        optimizer = torch.optim.Adam(clf.parameters(), 
                                     lr=0.01, 
                                     weight_decay=args.nodeclas_weight_decay)

        best_val_metric = test_metric = 0
        for epoch in tqdm(range(1, 101), desc=f'Training on runs {run}...'):
            clf.train()
            for nodes in train_loader:
                optimizer.zero_grad()
                loss_fn(clf(embedding[nodes]), y[nodes]).backward()
                optimizer.step()
                
            val_metric, test_metric = test(val_loader), test(test_loader)
            if val_metric > best_val_metric:
                best_val_metric = val_metric
                best_test_metric = test_metric
        results.append(best_test_metric)
        print(f'Runs {run}: accuracy {best_test_metric:.2%}')
                          
    print(f'Node Classification Results ({args.runs} runs):\n'
          f'Accuracy: {np.mean(results):.2%} ± {np.std(results):.2%}')
    
def centroid_distance(centroids1, centroids2):
    """
    Calculate the distance between two sets of centroids.
    input:
    centroids1: numpy array of shape (n_clusters, n_features)
    centroids2: numpy array of shape (n_clusters, n_features)
    return: 
    float, the distance between the two sets of centroids
    """
    if centroids1.shape != centroids2.shape:
        raise ValueError("The shape of the two sets of centroids must be the same.")
    distances = []
    centroids1 = centroids1.tolist()
    centroids2 = centroids2.tolist()

    while centroids1 and centroids2:
        min_distance = float('inf')
        min_pair = (None, None)
        
        for i, c1 in enumerate(centroids1):
            for j, c2 in enumerate(centroids2):
                distance = np.linalg.norm(np.array(c1) - np.array(c2))
                if distance < min_distance:
                    min_distance = distance
                    min_pair = (i, j)
        
        distances.append(min_distance)
        del centroids1[min_pair[0]]
        del centroids2[min_pair[1]]
    distance = np.mean(distances)
    return distance
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", nargs="?", default="Cora", help="Datasets. (default: Cora)")
parser.add_argument("--mask", nargs="?", default="Path", help="Masking stractegy, `Path`, `Edge` or `None` (default: Path)")













parser.add_argument('--seed', type=int, default=2022, help='Random seed for model and dataset. (default: 2022)')















parser.add_argument('--n_clients', type=int, default=10, help='number of clietns. (default: 4)')
parser.add_argument('--layer', type=str, default="gcn", help='GNN layer type, (default: gcn)')
parser.add_argument("--encoder_activation", nargs="?", default="elu", help="Activation function for GNN encoder, (default: elu)")
parser.add_argument('--encoder_channels', type=int, default=128, help='Channels of GNN encoder layers. (default: 128)')
parser.add_argument('--hidden_channels', type=int, default=64, help='Channels of hidden representation. (default: 64)')
parser.add_argument('--decoder_channels', type=int, default=32, help='Channels of decoder layers. (default: 128)')
parser.add_argument('--encoder_layers', type=int, default=2, help='Number of layers for encoder. (default: 2)')
parser.add_argument('--decoder_layers', type=int, default=2, help='Number of layers for decoders. (default: 2)')
parser.add_argument('--encoder_dropout', type=float, default=0.8, help='Dropout probability of encoder. (default: 0.8)')
parser.add_argument('--decoder_dropout', type=float, default=0.2, help='Dropout probability of decoder. (default: 0.2)')
parser.add_argument('--alpha', type=float, default=0., help='loss weight for degree prediction. (default: 0.)')

parser.add_argument('--lr', type=float, default=0.01, help='Learning rate for training. (default: 0.01)')
parser.add_argument('--weight_decay', type=float, default=5e-5, help='weight_decay for link prediction training. (default: 5e-5)')
parser.add_argument('--grad_norm', type=float, default=1.0, help='grad_norm for training. (default: 1.0.)')
parser.add_argument('--batch_size', type=int, default=2**16, help='Number of batch size for link prediction training. (default: 2**16)')

parser.add_argument("--start", nargs="?", default="node", help="Which Type to sample starting nodes for random walks, (default: node)")
parser.add_argument('--p', type=float, default=0.7, help='Mask ratio or sample ratio for MaskEdge/MaskPath')

parser.add_argument('--bn', action='store_true', help='Whether to use batch normalization for GNN encoder. (default: False)')
parser.add_argument('--l2_normalize', action='store_true', help='Whether to use l2 normalize output embedding. (default: False)')
parser.add_argument('--nodeclas_weight_decay', type=float, default=1e-3, help='weight_decay for node classification training. (default: 1e-3)')

parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs. (default: 500)')
parser.add_argument('--pre_epochs', type=int, default=100, help='Number of training epochs. (default: 100)')
parser.add_argument('--runs', type=int, default=10, help='Number of runs. (default: 10)')
parser.add_argument('--eval_period', type=int, default=50, help='(default: 30)')
parser.add_argument("--save_path", nargs="?", default="MaskGAE-NodeClas.pt", help="save path for model. (default: MaskGAE-NodeClas.pt)")
parser.add_argument("--device", type=int, default=0)
parser.add_argument('--full_data', action='store_true', help='Whether to use full data for pretraining. (default: False)')


try:
    args = parser.parse_args()
    print(args)
except:
    parser.print_help()
    exit(0)
set_seed(args.seed)
device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
dataset = Planetoid(root=os.path.join(os.path.dirname(__file__), 'cora'), name='Cora', transform=T.NormalizeFeatures())
data = dataset[0]
input_dim = data.x.shape[1]
n_nodes = data.x.shape[0]
# 打乱节点索引，划分成4个子图
indices = torch.randperm(n_nodes)
split_size = n_nodes // args.n_clients
splits = [indices[i * split_size:(i + 1) * split_size] for i in range(args.n_clients)]
client_list = []
for i in range(args.n_clients):
    client_data = data.clone().to(device)
    edge_index = data.edge_index.to(device) # 原始图的边索引
    split = splits[i].to(device)
    client_data.x = data.x.to(device)[split]
    client_data.y = data.y.to(device)[split]
    client_data.train_mask = data.train_mask.to(device)[split]
    client_data.val_mask = data.val_mask.to(device)[split]
    client_data.test_mask = data.test_mask.to(device)[split]
    client_data.edge_index = edge_index[:, torch.isin(edge_index[0], split) & torch.isin(edge_index[1], split)] # 客户端内部的边索引
    additional_edge_index = edge_index[:, (torch.isin(edge_index[0], split) & ~torch.isin(edge_index[1], split)) | (~torch.isin(edge_index[0], split) & torch.isin(edge_index[1], split))] # 客户端内节点与其他客户端节点连接的边
    additional_node_index = torch.unique(additional_edge_index[0,~torch.isin(additional_edge_index[0], split)]) # 来自其他客户端的全局节点索引，供后续边界传播使用
    original_node_index = torch.cat([split, additional_node_index]) # 内部节点和外部节点的索引
    client_data.aug_edge_index = torch.cat((client_data.edge_index,additional_edge_index),dim=1) # (2, edges_in+edges_additional)

    # 局部图索引对齐
    mapping = {global_idx.item(): local_idx for local_idx, global_idx in enumerate(original_node_index)}
    client_data.edge_index = torch.stack([
        torch.tensor([mapping[node.item()] for node in client_data.edge_index[0]]),
        torch.tensor([mapping[node.item()] for node in client_data.edge_index[1]])
    ])
    client_data.aug_edge_index = torch.stack([
        torch.tensor([mapping[node.item()] for node in client_data.aug_edge_index[0]]),
        torch.tensor([mapping[node.item()] for node in client_data.aug_edge_index[1]])
    ])
    client_model = MaskGAE(
        GNNEncoder(input_dim, args.encoder_channels, args.hidden_channels,
                   num_layers=args.encoder_layers, dropout=args.encoder_dropout,
                   bn=args.bn, layer=args.layer, activation=args.encoder_activation).to(device),
        EdgeDecoder(args.hidden_channels, args.decoder_channels,
                    num_layers=args.decoder_layers, dropout=args.decoder_dropout).to(device),
        DegreeDecoder(args.hidden_channels, args.decoder_channels,
                      num_layers=args.decoder_layers, dropout=args.decoder_dropout).to(device),
        MaskEdge(p=args.p)
    ).to(device)
    client_list.append(Client(i, client_data, client_model, original_node_index, device, use_model=False))

for client in client_list:
    #自编码器的训练
    client.Mask_GAE_train(args.pre_epochs)
    print(f"Client {client.id} GAE training finished.")
    # 训练完成后，获取每个客户端的嵌入
    client.embedding = client.MaskGAE_model(client.data.x, client.data.edge_index) #output shape (n_nodes, hidden_dim)
    # 计算每个客户端的嵌入的k-means聚类中心
    kmeans = KMeans(n_clusters=7, random_state=args.seed).fit(client.embedding.cpu().detach().numpy())
    client.centroids = kmeans.cluster_centers_ # shape (n_cluster, hidden_dim)

server = Server(client_list)
centroid = server.fed_kmeans() # fed_kmeans的聚类中心 (n_clusters, hidden_dim)
centroid = torch.tensor(centroid).to(device)
L=2
alpha = 0.2
v = 2
a = 0.1 # 损失函数超参数：L_clu系数
b = 0.2 # 损失函数超参数：L_gcn系数
torch.autograd.set_detect_anomaly(True)
maxQ = -1
cnt = 0
mods = []
for epoch in range(args.epochs):
        mod = 0
        node_embeddings1 = torch.zeros((n_nodes, args.hidden_channels), device=device)
        node_embeddings2 = torch.zeros((n_nodes, args.hidden_channels), device=device)
        node_embeddings3 = torch.zeros((n_nodes, args.hidden_channels), device=device)
        for client in client_list:
            client.Z1m = client.propagate_I(client.embedding) #input Z(l) (n_nodes, hidden_dim), output Z(l+0.5)(n_nodes, hidden_dim) deree_matrix (n_nodes+n_nodes_additional, n_nodes+n_nodes_additional)
            for local_idx, global_idx in enumerate(client.original_index[:client.n_nodes]):
                node_embeddings1[global_idx] = client.Z1m[local_idx].detach()
        # 从服务端获取虚拟节点信息h_additional(l+0.5)
            additional_embedding1 = node_embeddings1[client.original_index[client.n_nodes:]] # (n_nodes_additional, hidden_dim)
            client.Z1o = client.propagate_B(client.Z1m, additional_embedding1) #input Z(l+0.5) (n_nodes, hidden_dim) Z_additional(l+0.5)(n_nodes_additional), output Z(l+1)(n_nodes, hidden_dim)
            client.H1 = client.encoders[0](client.embedding)
            client.Z2i = (1-alpha) * client.Z1o + alpha * client.H1
            # Apply softmax to compute probabilities for each node belonging to each cluster
        for client in client_list:
            client.Z2m = client.propagate_I(client.Z2i) #input Z(l) (n_nodes, hidden_dim), output Z(l+0.5)(n_nodes, hidden_dim) deree_matrix (n_nodes+n_nodes_additional, n_nodes+n_nodes_additional)
            for local_idx, global_idx in enumerate(client.original_index[:client.n_nodes]):
                node_embeddings2[global_idx] = client.Z2m[local_idx].detach()
        # 从服务端获取虚拟节点信息h_additional(l+0.5)
            additional_embedding2 = node_embeddings2[client.original_index[client.n_nodes:]] # (n_nodes_additional, hidden_dim)
            client.Z2o = client.propagate_B(client.Z2m, additional_embedding2) #input Z(l+0.5) (n_nodes, hidden_dim) Z_additional(l+0.5)(n_nodes_additional), output Z(l+1)(n_nodes, hidden_dim)
            client.H2 = client.encoders[1](client.H1)
            client.Z3i = (1-alpha) * client.Z2o + alpha * client.H2
        for client in client_list:
            client.Z3m = client.propagate_I(client.Z3i) #input Z(l) (n_nodes, hidden_dim), output Z(l+0.5)(n_nodes, hidden_dim) deree_matrix (n_nodes+n_nodes_additional, n_nodes+n_nodes_additional)
            for local_idx, global_idx in enumerate(client.original_index[:client.n_nodes]):
                node_embeddings3[global_idx] = client.Z3m[local_idx].detach()
        # 从服务端获取虚拟节点信息h_additional(l+0.5)
            additional_embedding3 = node_embeddings3[client.original_index[client.n_nodes:]] # (n_nodes_additional, hidden_dim)
            client.Z3o = client.propagate_B(client.Z3m, additional_embedding3) #input Z(l+0.5) (n_nodes, hidden_dim) Z_additional(l+0.5)(n_nodes_additional), output Z(l+1)(n_nodes, hidden_dim)
            client.H3 = client.encoders[2](client.H2)
            client.Z = (1-alpha) * client.Z3o + alpha * client.H3
            client.Z_out = client.Z_fc(client.Z)
            client.Z_probabilities = F.softmax(client.Z_out, dim=1)  # output Shape: (num_nodes, n_clusters)
            client.Q = (1+torch.cdist(client.H3, centroid)/v)**(-(v+1)/2) # Q Shape: (num_nodes, n_clusters)
            div_q = torch.sum(client.Q,dim=1) #shape (num_nodes)
            client.Q = client.Q / div_q.view(-1, 1) # Q Shape: (num_nodes, n_clusters)
            f = torch.sum(client.Q, dim=0)  #f shape: (n_clusters)
            f[f==0] = 1e-8 # f shape: (n_clusters)
            client.P = (client.Q ** 2) / f.view(1, -1) # P Shape: (num_nodes, n_clusters)
            div_p = torch.sum(client.P,dim=1) #shape (num_nodes)
            client.P = client.P / div_p.view(-1, 1) # P Shape: (num_nodes, n_clusters)
            client.L_clu = F.kl_div(client.Q.log(), client.P, reduction='sum')
            client.L_gcn =F.kl_div(client.Z_probabilities.log(), client.P, reduction='sum')
            rebuild_H = client.rebuild(client.H3)
            client.L_res = F.mse_loss(client.embedding,rebuild_H)
            client.mod = modularity(client.Z_probabilities.argmax(dim=1), client.data.edge_index)
            client.loss = client.L_res + a * client.L_clu + b * client.L_gcn
            mod = mod + client.mod
        mod = mod / args.n_clients
        mods.append(mod)
        loss = server.avg_loss()
        print(f"Epoch {epoch}, Loss: {loss:.4f} Q: {mod:.5f}")
        if mod >= maxQ:
            cnt = 0
            maxQ = mod
        else:
            cnt += 1
        if cnt >= 8:
            print(f"bestQ: {maxQ}")
            break
        server.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        server.optimizer.step()
        if epoch % args.eval_period == 0:
            # 收集所有客户端的预测标签
            global_labels = torch.zeros(n_nodes, dtype=torch.long, device=device)
            with torch.no_grad():
                node_embeddings1 = torch.zeros((n_nodes, args.hidden_channels), device=device)
                node_embeddings2 = torch.zeros((n_nodes, args.hidden_channels), device=device)
                node_embeddings3 = torch.zeros((n_nodes, args.hidden_channels), device=device)
                for client in client_list:
                    client.Z1m = client.propagate_I(client.embedding) #input Z(l) (n_nodes, hidden_dim), output Z(l+0.5)(n_nodes, hidden_dim) deree_matrix (n_nodes+n_nodes_additional, n_nodes+n_nodes_additional)
                    for local_idx, global_idx in enumerate(client.original_index[:client.n_nodes]):
                        node_embeddings1[global_idx] = client.Z1m[local_idx].detach()
                # 从服务端获取虚拟节点信息h_additional(l+0.5)
                    additional_embedding1 = node_embeddings1[client.original_index[client.n_nodes:]] # (n_nodes_additional, hidden_dim)
                    client.Z1o = client.propagate_B(client.Z1m, additional_embedding1) #input Z(l+0.5) (n_nodes, hidden_dim) Z_additional(l+0.5)(n_nodes_additional), output Z(l+1)(n_nodes, hidden_dim)
                    client.H1 = client.encoders[0](client.embedding)
                    client.Z2i = (1-alpha) * client.Z1o + alpha * client.H1
                    # Apply softmax to compute probabilities for each node belonging to each cluster
                for client in client_list:
                    client.Z2m = client.propagate_I(client.Z2i) #input Z(l) (n_nodes, hidden_dim), output Z(l+0.5)(n_nodes, hidden_dim) deree_matrix (n_nodes+n_nodes_additional, n_nodes+n_nodes_additional)
                    for local_idx, global_idx in enumerate(client.original_index[:client.n_nodes]):
                        node_embeddings2[global_idx] = client.Z2m[local_idx].detach()
                # 从服务端获取虚拟节点信息h_additional(l+0.5)
                    additional_embedding2 = node_embeddings2[client.original_index[client.n_nodes:]] # (n_nodes_additional, hidden_dim)
                    client.Z2o = client.propagate_B(client.Z2m, additional_embedding2) #input Z(l+0.5) (n_nodes, hidden_dim) Z_additional(l+0.5)(n_nodes_additional), output Z(l+1)(n_nodes, hidden_dim)
                    client.H2 = client.encoders[1](client.H1)
                    client.Z3i = (1-alpha) * client.Z2o + alpha * client.H2
                for client in client_list:
                    client.Z3m = client.propagate_I(client.Z3i) #input Z(l) (n_nodes, hidden_dim), output Z(l+0.5)(n_nodes, hidden_dim) deree_matrix (n_nodes+n_nodes_additional, n_nodes+n_nodes_additional)
                    for local_idx, global_idx in enumerate(client.original_index[:client.n_nodes]):
                        node_embeddings3[global_idx] = client.Z3m[local_idx].detach()
                # 从服务端获取虚拟节点信息h_additional(l+0.5)
                    additional_embedding3 = node_embeddings3[client.original_index[client.n_nodes:]] # (n_nodes_additional, hidden_dim)
                    client.Z3o = client.propagate_B(client.Z3m, additional_embedding3) #input Z(l+0.5) (n_nodes, hidden_dim) Z_additional(l+0.5)(n_nodes_additional), output Z(l+1)(n_nodes, hidden_dim)
                    client.H3 = client.encoders[2](client.H2)
                    client.Z = (1-alpha) * client.Z3o + alpha * client.H3
                    client.Z_out = client.Z_fc(client.Z)
                    client.Z_probabilities = F.softmax(client.Z_out, dim=1)  # output Shape: (num_nodes, n_clusters)
                    predicted_labels = torch.argmax(client.Z_probabilities, dim=1)
                    for local_idx, global_idx in enumerate(client.original_index[:client.n_nodes]):
                        global_labels[global_idx] = predicted_labels[local_idx]
                # 计算modularity
                modularity_score = modularity(global_labels, edge_index)
                print(f"Epoch: {epoch} Modularity score: {modularity_score}")

# 绘制mods的变化曲线
plt.plot(range(0, len(mods)), mods)
plt.xlabel('Epoch')
plt.ylabel('Modularity')
xticks = [i for i in range(0, len(mods), 5)]
plt.xticks(xticks)
plt.title(f'n_clients = {args.n_clients}')
plt.show()