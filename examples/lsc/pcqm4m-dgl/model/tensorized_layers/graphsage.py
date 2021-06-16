import torch
from torch import nn as nn
from torch.nn import functional as F

import torchbnn as bnn
import torch.optim as optim

class BatchedGraphSAGE(nn.Module):
    def __init__(self, infeat, outfeat, use_bn=True,
                 mean=False, add_self=False):
        super().__init__()
        self.add_self = add_self
        self.use_bn = use_bn
        self.mean = mean
        self.W = nn.Linear(infeat, outfeat, bias=True)

        nn.init.xavier_uniform_(
            self.W.weight,
            gain=nn.init.calculate_gain('relu'))

    def forward(self, x, adj):
        num_node_per_graph = adj.size(1)
        if self.use_bn and not hasattr(self, 'bn'):
            self.bn = nn.BatchNorm1d(num_node_per_graph).to(adj.device)

        if self.add_self:
            adj = adj + torch.eye(num_node_per_graph).to(adj.device)

        if self.mean:
            adj = adj / adj.sum(-1, keepdim=True)

        h_k_N = torch.matmul(adj, x)
        h_k = self.W(h_k_N)
        h_k = F.normalize(h_k, dim=2, p=2)
        h_k = F.relu(h_k)
        if self.use_bn:
            h_k = self.bn(h_k)
        return h_k

    def __repr__(self):
        if self.use_bn:
            return 'BN' + super(BatchedGraphSAGE, self).__repr__()
        else:
            return super(BatchedGraphSAGE, self).__repr__()

class BayesBatchedGraphSAGE(BatchedGraphSAGE):
    def __init__(self, infeat, outfeat, use_bn=True,
                 mean=False, add_self=False):
        super().__init__(infeat, outfeat, use_bn,
                 mean, add_self)
        self.W = bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=infeat, out_features=outfeat)
