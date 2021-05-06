import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import global_add_pool

import pandas as pd
from tqdm import tqdm
from torch_geometric.data import DataLoader
import numpy as np
import random

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


"""
dynamic triplet dataset based on error
"""
# def createTripletLoader(model, train_loader, dataset, errorThres = 5) -> (anchor_loader, pos_loader, neg_loader)
def createTripletLoader(device, model, train_loader, dataset, args, errorThres = 5):

    # 2. get losses for training dataset
    y_true, y_pred = [], []
    for step, batch in enumerate(tqdm(train_loader, desc="Sampling Triplets")):
        # put batch to cuda
        batch = batch.to(device)

        # get prediction
        pred = model(batch).view(-1, )
        pred = pred.detach().cpu().tolist()
        y_pred.extend(pred)

        # get labels
        label = batch.y.detach().cpu().tolist()
        y_true.extend(label)
        
        if step == 100:
            break


    # 3. convert to dataframe
    trainDF = pd.DataFrame(zip(y_pred, y_true), columns=["y_pred", "y_true"])
    trainDF["error"] = (trainDF["y_pred"] - trainDF["y_true"]).apply(lambda x: abs(x))
    # bin y_pred
    trainDF["y_class"] = trainDF["y_true"].apply(lambda x: int(np.floor(x)))

    # 4. pick data with error threshold < k
    highErrorDF = trainDF[trainDF.error > errorThres]
    lowErrorDF = trainDF[trainDF.error < errorThres]
    # create [anchorID, posId, negID]
    triplets = []
    # get number of data
    ndata = len(y_pred)
    for i, row in tqdm(list(highErrorDF.iterrows())):
        i_class = row["y_class"]

        # 4a. set i to be pos, find anchor and neg samples
        # set default to be random
        tripA = [random.randint(0, ndata), random.randint(0, ndata), random.randint(0, ndata)]
        tripA[1] = i
        # find anchor by sampling from lowErrorDF of same class
        tripA[0] = lowErrorDF[lowErrorDF.y_class == i_class].sample(1).index.item()
        # find negative by sampling from lowErrorDF of other class
        tripA[2] = lowErrorDF[lowErrorDF.y_class != i_class].sample(1).index.item()
        triplets.append(tripA)

        # 4b. set i to be neg, find anchor and pos samples
        # set default to be random
        tripB = [random.randint(0, ndata), random.randint(0, ndata), random.randint(0, ndata)]
        tripB[2] = i
        # find anchor by sampling from lowErrorDF of same class
        tripB[0] = lowErrorDF[lowErrorDF.y_class != i_class].sample(1).index.item()
        # find positive by sampling from lowErrorDF of other class
        tripB[1] = lowErrorDF[lowErrorDF.y_class != i_class].sample(1).index.item()
        triplets.append(tripB)

    if len(triplets) == 0:
        raise Exception("no triplets found")
        
    # 5. create anchor, pos and neg IDs
    triplets = np.array(triplets)
    anchorIDs = list(triplets[:, 0])
    posIDs = list(triplets[:, 1])
    negIDs = list(triplets[:, 2])

    # 6. create triplet loaders
    anchor_loader = DataLoader(dataset[anchorIDs], batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
    positive_loader = DataLoader(dataset[posIDs], batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
    negative_loader = DataLoader(dataset[negIDs], batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)

    return anchor_loader, positive_loader, negative_loader