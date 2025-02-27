import torch
import torch.nn as nn

from dgl.nn.pytorch import SumPooling, AvgPooling, MaxPooling, GlobalAttentionPooling, Set2Set

from conv import GNN_node, GNN_node_Virtualnode

class GNN(nn.Module):

    def __init__(self, num_tasks = 1, num_layers = 5, emb_dim = 300, gnn_type = 'gin',
                 virtual_node = True, residual = False, drop_ratio = 0, JK = "last",
                 graph_pooling = "sum"):
        '''
            num_tasks (int): number of labels to be predicted
            virtual_node (bool): whether to add virtual node or not
        '''
        super(GNN, self).__init__()

        self.num_layers = num_layers
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.graph_pooling = graph_pooling

        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ### GNN to generate node embeddings
        if virtual_node:
            self.gnn_node = GNN_node_Virtualnode(num_layers, emb_dim, JK = JK,
                                                 drop_ratio = drop_ratio,
                                                 residual = residual,
                                                 gnn_type = gnn_type)
        else:
            self.gnn_node = GNN_node(num_layers, emb_dim, JK = JK, drop_ratio = drop_ratio,
                                     residual = residual, gnn_type = gnn_type)


        ### Pooling function to generate whole-graph embeddings
        if self.graph_pooling == "sum":
            self.pool = SumPooling()
        elif self.graph_pooling == "mean":
            self.pool = AvgPooling()
        elif self.graph_pooling == "max":
            self.pool = MaxPooling
        elif self.graph_pooling == "attention":
            self.pool = GlobalAttentionPooling(
                gate_nn = nn.Sequential(nn.Linear(emb_dim, 2*emb_dim),
                                        nn.BatchNorm1d(2*emb_dim),
                                        nn.ReLU(),
                                        nn.Linear(2*emb_dim, 1)))

        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, n_iters = 2, n_layers = 2)
        else:
            raise ValueError("Invalid graph pooling type.")

        if graph_pooling == "set2set":
            self.graph_pred_linear = nn.Linear(2*self.emb_dim, self.num_tasks)
        else:
            self.graph_pred_linear = nn.Linear(self.emb_dim, self.num_tasks)

    def forward(self, g, x, edge_attr):
        h_node = self.gnn_node(g, x, edge_attr)

        h_graph = self.pool(g, h_node)
        output = self.graph_pred_linear(h_graph)

        if self.training:
            return output
        else:
            return torch.clamp(output, min=0, max=50)

"""
DiffPool
based on https://github.com/dmlc/dgl/tree/master/examples/pytorch/diffpool
two hierarchical layers
"""
from model.dgl_layers import GraphSage, GraphSageLayer, DiffPoolBatchedGraphLayer, BayesDiffPoolBatchedGraphLayer
from model.tensorized_layers import BatchedGraphSAGE, BayesBatchedGraphSAGE
from model.model_utils import batch2tensor
from torch.nn import init
import torch.nn.functional as F

class DiffPoolGNN(GNN):
    def __init__(self, num_tasks = 1, num_layers = 5, emb_dim = 300, gnn_type = 'gin',
                 virtual_node = True, residual = False, drop_ratio = 0, JK = "last",
                 graph_pooling = "sum"):
        super(DiffPoolGNN, self).__init__(num_tasks, num_layers, emb_dim, gnn_type,
                 virtual_node, residual, drop_ratio, JK, graph_pooling)
               
        # 2x number of outputs
        self.graph_pred_linear = nn.Linear(2*self.emb_dim, self.num_tasks)
#         self.graph_pred_linear = nn.Linear(self.emb_dim, self.num_tasks)

        self.first_diffpool_layer = DiffPoolBatchedGraphLayer(
            input_dim=600, # graph embedding dimension
            assign_dim=5, # group to 10
            output_feat_dim=600,
            activation=F.relu,
            dropout=0.0,
            aggregator_type="meanpool",
            link_pred=False
        ) #.to(device)

        self.gc_after_pool = BatchedGraphSAGE(600, 600) #.to(device)

    def forward(self, g, x, edge_attr):
        # 1. GCN: 3628x9 -> 3628x600
        g.ndata['h'] = x
        h_node = self.gnn_node(g, x, edge_attr)
#         print(h_node.shape)

        # 2. Graph Pooling 256x600
        h_graph_1 = self.pool(g, h_node)
#         print(h_graph.shape)
        
        # 3. DiffPool: (1280x1280), (1280x600)
        adj, h_node = self.first_diffpool_layer(g, h_node)
#         print(adj.shape, h_node.shape)
        
        # 3b. split to batches
        node_per_pool_graph = int(adj.size()[0] / len(g.batch_num_nodes()))
        h_node, adj = batch2tensor(adj, h_node, node_per_pool_graph)
#         print(adj.shape, h_node.shape)

        # 4. GCN:
        h_node = self.gcn_forward_tensorized(h_node, adj, [self.gc_after_pool], True)
#         print(h_node.shape)

        # 5. Graph Pooling 256x600
        h_graph_2 = torch.sum(h_node, dim=1)
#         print("h_graph_2", h_graph_2.shape)

        # 6. Last Layer; Combine Graph Embeddings
#         print(h_graph_1.shape, h_graph_2.shape)
        h_graph = torch.cat([h_graph_1, h_graph_2], dim=1)
#         h_graph = h_graph_1

        output = self.graph_pred_linear(h_graph)
#         print("output", output.shape)

        if self.training:
            return output
        else:
            return torch.clamp(output, min=0, max=50)

    def gcn_forward_tensorized(self, h, adj, gc_layers, cat=False):
        block_readout = []
        for gc_layer in gc_layers:
            h = gc_layer(h, adj)
            block_readout.append(h)
        if cat:
            block = torch.cat(block_readout, dim=2)  # N x F, F = F1 + F2 + ...
        else:
            block = h
        return block
    
    
import torchbnn as bnn
import torch.optim as optim

class BayesDiffPoolGNN(DiffPoolGNN):
    def __init__(self, num_tasks = 1, num_layers = 5, emb_dim = 300, gnn_type = 'gin',
                 virtual_node = True, residual = False, drop_ratio = 0, JK = "last",
                 graph_pooling = "sum"):
        super(BayesDiffPoolGNN, self).__init__(num_tasks, num_layers, emb_dim, gnn_type,
                 virtual_node, residual, drop_ratio, JK, graph_pooling)
        
        # 2x number of outputs
        self.graph_pred_linear = torch.nn.Sequential(
            bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=2*self.emb_dim, out_features=100),
            torch.nn.ReLU(),
            bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=100, out_features=100),
            torch.nn.ReLU(),
            bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=100, out_features=100),
            torch.nn.ReLU(),
            bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=100, out_features=self.num_tasks),
        )
        
        self.first_diffpool_layer = BayesDiffPoolBatchedGraphLayer(
            input_dim=600, # graph embedding dimension
            assign_dim=5, # group to 10
            output_feat_dim=600,
            activation=F.relu,
            dropout=0.0,
            aggregator_type="meanpool",
            link_pred=False
        )

        self.gc_after_pool = BayesBatchedGraphSAGE(600, 600)

        # KL-divergence loss for Bayesian Neural Network
        self.kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)
        self.kl_weight = 1 # 0.01

    def get_kl_loss(self):
        kl_losses = self.kl_loss(self.graph_pred_linear) + self.kl_loss(self.first_diffpool_layer) + self.kl_loss(self.gc_after_pool)
        return self.kl_weight * kl_losses