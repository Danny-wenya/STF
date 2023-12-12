import os
import pandas as pd
import numpy as np
import pickle
import torch
import time
from torch.autograd import Variable

from data.dataset.multi_step_dataset import MultiStepDataset

def gen_data(data, ntr, N):
    '''
    if flag:
        data=pd.read_csv(fname)
    else:
        data=pd.read_csv(fname,header=None)
    '''
    #data=data.as_matrix()
    data=data[:180*(data.shape[0]//180)]
    data=np.reshape(data,[-1,180,N])
    return data

def normalize(a):
    mu=np.mean(a,axis=1,keepdims=True)
    std=np.std(a,axis=1,keepdims=True)
    return (a-mu)/std

def compute_dtw(a,b,order=1,Ts=12,normal=True):
    if normal:
        a=normalize(a) #（54，288）
        b=normalize(b) #（54，288）
    T0=a.shape[1] #288
    # （54，288，288）：（54，1，288）-（54，288，1）=形成了广播，使得a每一天的数据和b所有天的数据做差，最终维度（54，288，288）
    d=np.reshape(a,[-1,1,T0])-np.reshape(b,[-1,T0,1])  
    # 计算范数在计算相似度矩阵时可以用于标准化或规范化向量，以便更好地度量它们之间的相似性。
    d=np.linalg.norm(d,axis=0,ord=order) #计算列的L1范数
    D=np.zeros([T0,T0]) #初始化相似度矩阵
    for i in range(T0): #遍历一天的数据，将每一条的数据与其后12条数据做处理，以12为窗口长度滑动   这是什么算法？？？？？？
        for j in range(max(0,i-Ts),min(T0,i+Ts+1)): # 每次截取长度为12的数据进行计算，根据a,b天数的不同，将差值矩阵值进行变形。
            if (i==0) and (j==0): # 第0天的数据特殊处理
                D[i,j]=d[i,j]**order # 第0天本身    此相似矩阵应该用图画出更加直观。
                continue
            if (i==0): 
                D[i,j]=d[i,j]**order+D[i,j-1] # 当i==0，a矩阵第0天的数据均与变形后的前一天的数据相加。达到在时间序列相互包含的目的。
                continue
            if (j==0):
                D[i,j]=d[i,j]**order+D[i-1,j]
                continue
            if (j==i-Ts): # 当j
                D[i,j]=d[i,j]**order+min(D[i-1,j-1],D[i-1,j])
                continue
            if (j==i+Ts):
                D[i,j]=d[i,j]**order+min(D[i-1,j-1],D[i,j-1])
                continue
            D[i,j]=d[i,j]**order+min(D[i-1,j-1],D[i-1,j],D[i,j-1])
    return D[-1,-1]**(1.0/order)

def construct_adj_fusion(A, A_dtw, steps):
    '''
    construct a bigger adjacency matrix using the given matrix

    Parameters
    ----------
    A: np.ndarray, adjacency matrix, shape is (N, N)

    steps: how many times of the does the new adj mx bigger than A

    Returns
    ----------
    new adjacency matrix: csr_matrix, shape is (N * steps, N * steps)

    ----------
    This is 4N_1 mode:

    [T, 1, 1, T
     1, S, 1, 1
     1, 1, S, 1
     T, 1, 1, T]

    '''

    N = len(A)
    adj = np.zeros([N * steps] * 2) # "steps" = 4 !!!

    for i in range(steps):
        if (i == 1) or (i == 2):
            adj[i * N: (i + 1) * N, i * N: (i + 1) * N] = A
        else:
            adj[i * N: (i + 1) * N, i * N: (i + 1) * N] = A_dtw
    #'''
    for i in range(N):
        for k in range(steps - 1):
            adj[k * N + i, (k + 1) * N + i] = 1
            adj[(k + 1) * N + i, k * N + i] = 1
    #'''
    adj[3 * N: 4 * N, 0:  N] = A_dtw #adj[0 * N : 1 * N, 1 * N : 2 * N]
    adj[0 : N, 3 * N: 4 * N] = A_dtw #adj[0 * N : 1 * N, 1 * N : 2 * N]

    adj[2 * N: 3 * N, 0 : N] = adj[0 * N : 1 * N, 1 * N : 2 * N]
    adj[0 : N, 2 * N: 3 * N] = adj[0 * N : 1 * N, 1 * N : 2 * N]
    adj[1 * N: 2 * N, 3 * N: 4 * N] = adj[0 * N : 1 * N, 1 * N : 2 * N]
    adj[3 * N: 4 * N, 1 * N: 2 * N] = adj[0 * N : 1 * N, 1 * N : 2 * N]


    for i in range(len(adj)):
        adj[i, i] = 1

    return adj


class STFGNNDataset(MultiStepDataset):

    def __init__(self, config):
        super().__init__(config)
        self.strides = self.config.get("strides", 4)
        self.order = self.config.get("order", 1)
        self.lag = self.config.get("lag", 12)
        self.period = self.config.get("period", 288)
        self.sparsity = self.config.get("sparsity", 0.01)
        self.train_rate = self.config.get("train_rate", 0.6)
        self.adj_mx = torch.FloatTensor(self._construct_adj())
        # self.adj_mx = torch.randn((1432, 1432))


    def _construct_dtw(self):
        data = self.rawdat[:, :, 0]
        N=data.shape[1]
        data=data[:(data.shape[0]//180)*180,...]
        xtr=data.reshape(-1,180,N)
        d = np.zeros([N, N])
        for i in range(N):
            for j in range(i+1,N):
                d[i,j]=compute_dtw(xtr[:,:,i],xtr[:,:,j])

        print("The calculation of time series is done!")
        dtw = d+ d.T
        n = dtw.shape[0]
        w_adj = np.zeros([n,n])
        adj_percent = 0.01
        top = int(n * adj_percent)
        for i in range(dtw.shape[0]):
            a = dtw[i,:].argsort()[0:top]  # 将i路口与其他所有路口的相似度进行排序，选择相似度最高的top个路口。
            for j in range(top):
                w_adj[i, a[j]] = 1 #将被选择的路口在邻接矩阵中置1.
        
        for i in range(n):
            for j in range(n):
                if (w_adj[i][j] != w_adj[j][i] and w_adj[i][j] ==0):
                    w_adj[i][j] = 1
                if( i==j):
                    w_adj[i][j] = 1

        print("Total route number: ", n)
        print("Sparsity of adj: ", len(w_adj.nonzero()[0])/(n*n))
        print("The weighted matrix of temporal graph is generated!")
        self.dtw = w_adj # 相似矩阵中将每个路口的相似路口置1.


    def _construct_adj(self):
        """
        构建local 时空图
        :param A: np.ndarray, adjacency matrix, shape is (N, N)
        :param steps: 选择几个时间步来构建图
        :return: new adjacency matrix: csr_matrix, shape is (N * steps, N * steps)
        """
        self._construct_dtw() #得到self.dtw
        adj_mx = construct_adj_fusion(self.adj_mx, self.dtw, self.strides)  #self.adj_mx = torch.FloatTensor(self._construct_adj()),这算不算递归？self.strides=4
        print("The shape of localized adjacency matrix: {}".format(
        adj_mx.shape), flush=True)

        return adj_mx


    def get_data(self):
        """
        返回数据的DataLoader，包括训练数据、测试数据、验证数据

        Returns:
            tuple: tuple contains:
                train_dataloader:
                eval_dataloader:
                test_dataloader:
        """
        # 加载数据集

        return self.data["train_loader"], self.data["valid_loader"], self.data["test_loader"]

    def get_data_feature(self):
        """
        返回数据集特征，子类必须实现这个函数，返回必要的特征

        Returns:
            dict: 包含数据集的相关特征的字典
        """
        feature = {
            "scaler": self.data["scaler"],
            "adj_mx": self.adj_mx,
            "num_batches": self.data['num_batches']
        }

        return feature









