import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import global_add_pool

class TripletLossRegression(nn.Module):
    """
        anchor, positive, negative are node-level embeddings of a GNN before they are sent to a pooling layer,
        and hence are expected to be matrices.
        anchor_gt, positive_gt, and negative_gt are ground truth tensors that correspond to the ground-truth
        values of the anchor, positive, and negative respectively.
    """

    def __init__(self, margin: float = 0.0, eps=1e-6):
        super(TripletLossRegression, self).__init__()
        self.margin = margin
        self.eps = eps

    def forward(self, anchor_batch, negative_batch, positive_batch,
                anchor: Tensor, negative: Tensor, positive: Tensor,
                anchor_gt: Tensor, negative_gt: Tensor, positive_gt: Tensor) -> Tensor:
        anchor = global_add_pool(anchor, anchor_batch)

        positive = global_add_pool(positive, positive_batch)

        negative = global_add_pool(negative, negative_batch)

        pos_distance = torch.linalg.norm(positive - anchor, dim=1)
        negative_distance = torch.linalg.norm(negative - anchor, dim=1)

        coeff = torch.div(torch.abs(negative_gt - anchor_gt) , (torch.abs(positive_gt - anchor_gt) + self.eps))
        loss = F.relu((pos_distance - coeff * negative_distance) + self.margin)
        return torch.mean(loss)


