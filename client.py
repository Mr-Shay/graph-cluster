import torch
import math
from notfed import DSCN
class Client:
    def __init__(self, id, data, MaskGAE_model, original_index, device, use_model =True):
        self.id = id
        self.data = data.to(device) 
        '''
        data.x (n_nodes, feature_dim)
        data.edge_index (2, n_edges)
        data.aug_edge_index (2, n_edges+n_edges_additional) # 包含虚拟节点的边
        data.y data.train_mask data.test_mask data.val_mask (n_nodes,)
        '''
        self.MaskGAE_model = MaskGAE_model
        self.original_index = original_index # 在原始图中的内部和虚拟节点索引
        self.n_nodes = data.x.shape[0] # 内部节点数目
        if use_model:
            self.model = DSCN(device)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        else:
            encoder1 = torch.nn.Sequential(
                torch.nn.Linear(64, 64),
                torch.nn.Sigmoid()
            ).to(device)
            encoder2 = torch.nn.Sequential(
                torch.nn.Linear(64, 64),
                torch.nn.Sigmoid()
            ).to(device)
            encoder3 = torch.nn.Sequential(
                torch.nn.Linear(64, 64),
                torch.nn.Sigmoid()
            ).to(device)
            decoder1 = torch.nn.Sequential(
                torch.nn.Linear(64, 64),
                torch.nn.Sigmoid()
            ).to(device)
            decoder2 = torch.nn.Sequential(
                torch.nn.Linear(64, 64),
                torch.nn.Sigmoid()
            ).to(device)
            decoder3 = torch.nn.Sequential(
                torch.nn.Linear(64, 64),
                torch.nn.Sigmoid()
            ).to(device)
            self.encoders = [encoder1, encoder2, encoder3]
            self.decoders = [decoder1, decoder2, decoder3]
            self.Z_fc = torch.nn.Linear(64,7).to(device)

        self.device = device

    def Mask_GAE_train(self, epochs=100, lr=0.001, weight_decay=5e-5, earlystop=True):
        self.MaskGAE_model.train()
        optimizer = torch.optim.Adam([{'params':self.MaskGAE_model.encoder.parameters()},
                              {'params':self.MaskGAE_model.edge_decoder.parameters()},
                              {'params':self.MaskGAE_model.degree_decoder.parameters()}],
                              lr=lr, weight_decay=weight_decay)
        cnt = 0
        min_loss = float('inf')
        for i in range(epochs):
            gaeloss = self.MaskGAE_model.train_step(self.data,optimizer)
            if earlystop:
                if gaeloss < min_loss:
                    min_loss = gaeloss
                    cnt = 0
                else:
                    cnt += 1
                    if cnt > 20:
                        break # Early stopping
            #print(f"Epoch {i}, Loss: {gaeloss:.4f}")

    def propagate_I(self, h):
        """
        内部传播
        input:
            h: (n_nodes, feature_dim) 内部节点的隐层表示
        output:
            degree_matrix (n_nodes+n_nodes_additional, n_nodes+n_nodes_additional) 传给外部传播模块继续使用
            h' (n_nodes, feature_dim) h(l+0.5)
        """
        n_nodes_additional = self.original_index.shape[0] - self.n_nodes # 虚拟节点数
        feature_dim = h.shape[1]
        aug_h = torch.cat((h, torch.zeros(n_nodes_additional, feature_dim).to(h.device)), dim=0) # (n_nodes+n_nodes_additional, feature_dim) aug_h是添加了虚拟节点的隐层表示，虚拟节点的隐层表示为0
        edge_index_I = self.data.aug_edge_index # (2, n_edges+n_edges_additional) 包含虚拟节点的边
        degrees = torch.bincount(edge_index_I[0]) # 每个节点的度数 (n_nodes+n_nodes_additional,)   
        degrees = (1+degrees)**(-0.5)
        self.degree_matrix = torch.diag(degrees).to(self.device) # (n_nodes+n_nodes_additional, n_nodes+n_nodes_additional) 度矩阵
        adjency_matrix = torch.zeros((self.original_index.shape[0],self.original_index.shape[0])).to(self.device) # (n_nodes+n_nodes_additional, n_nodes+n_nodes_additional) 邻接矩阵
        adjency_matrix[edge_index_I[0], edge_index_I[1]] = 1 # 邻接矩阵的元素为1
        # 计算D矩阵，由邻接矩阵乘度矩阵得到
        D = torch.matmul(adjency_matrix, self.degree_matrix) # D shape: (n_nodes+n_nodes_additional, n_nodes+n_nodes_additional)
        output_h = torch.matmul(D, aug_h)
        output_h = output_h[:self.n_nodes, :] # output_h shape: (n_nodes, hidden_dim)
        return output_h



    def propagate_B(self, h, h_additional):
        """
        边界传播
        input:
            h: (n_nodes, feature_dim) 内部节点的隐层表示 h(l+0.5)
            h_additional: (n_nodes_additional, feature_dim) 接收到的虚拟节点的隐层表示 h_addtional(l+0.5)

        output:
            h' (n_nodes, feature_dim) h(l+1)
        """
        edge_index_B = self.data.aug_edge_index[:,self.data.edge_index.shape[1]:] #去除所有内部边
        adjency_matrix = torch.eye(self.original_index.shape[0]).to(self.device) # (n_nodes+n_nodes_additional, n_nodes+n_nodes_additional) 邻接矩阵初始化，带自环
        adjency_matrix[edge_index_B[0], edge_index_B[1]] = 1 # 邻接矩阵的元素为1
        degree_matrix = self.degree_matrix.detach()
        D = torch.matmul(degree_matrix, adjency_matrix) # D shape: (n_nodes+n_nodes_additional, n_nodes+n_nodes_additional) degree_matrix由内部传播时初始化并计算
        output_h = torch.matmul(D, torch.cat((h, h_additional), dim=0)) # output_h shape: (n_nodes+n_nodes_additional, hidden_dim)
        output_h = output_h[:self.n_nodes, :] # output_h shape: (n_nodes, hidden_dim)
        return output_h

    
    def rebuild(self,H):
        H1 = self.decoders[0](H)
        H2 = self.decoders[1](H1)
        H3 = self.decoders[2](H2)
        return H3