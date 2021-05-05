from ogb.lsc import PygPCQM4MDataset, PCQM4MEvaluator

import torch
from torch_geometric.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR

from gnn import GNN, BayesianGNN

import os
from tqdm import tqdm
import argparse
import numpy as np
import random

from loss_functions.TripletLossRegression import TripletLossRegression

reg_criterion = torch.nn.L1Loss()
model_activation = {}
def get_activation(name):
    def hook(model, input, output):
        model_activation[name] = output
    return hook

def triplet_loss_train(model, device, anchor_loader, negative_loader, positive_loader, optimizer, gnn_name):
    model.train()
    loss_accum = 0
    triplet_loss_criterion = TripletLossRegression()

    for step, (anchor_batch, negative_batch, positive_batch) in \
            enumerate(zip(tqdm(anchor_loader, desc="Iteration"), negative_loader, positive_loader)):
        anchor_batch = anchor_batch.to(device)
        pred_anchor = model(anchor_batch).view(-1,)
        anchor_embed = model_activation['gnn_node']

        negative_batch = negative_batch.to(device)
        pred_neg = model(negative_batch).view(-1,)
        neg_embed = model_activation['gnn_node']

        positive_batch = positive_batch.to(device)
        pred_pos= model(positive_batch).view(-1,)
        pos_embed = model_activation['gnn_node']

        optimizer.zero_grad()
        mae_loss = reg_criterion(pred_anchor, anchor_batch.y)
        tll_loss = triplet_loss_criterion(anchor_batch.batch, negative_batch.batch, positive_batch.batch,
                                          anchor_embed, neg_embed, pos_embed,
                                          anchor_batch.y, negative_batch.y, positive_batch.y)
        loss = mae_loss + tll_loss

        if gnn_name == 'gin-virtual-bnn':
            kl_loss = model.get_kl_loss()[0]
            loss += kl_loss

        loss.backward()
        optimizer.step()

        loss_accum += loss.detach().cpu().item()

    return loss_accum / (step + 1)

def train(model, device, loader, optimizer, gnn_name):
    model.train()
    loss_accum = 0

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        pred = model(batch).view(-1,)
        optimizer.zero_grad()
        loss = reg_criterion(pred, batch.y)

        if gnn_name == 'gin-virtual-bnn':
            kl_loss = model.get_kl_loss()[0]
            loss += kl_loss

        loss.backward()
        optimizer.step()

        loss_accum += loss.detach().cpu().item()

    return loss_accum / (step + 1)

def eval(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch).view(-1,)

        y_true.append(batch.y.view(pred.shape).detach().cpu())
        y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim = 0)
    y_pred = torch.cat(y_pred, dim = 0)

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)["mae"]

def test(model, device, loader):
    model.eval()
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch).view(-1,)

        y_pred.append(pred.detach().cpu())

    y_pred = torch.cat(y_pred, dim = 0)

    return y_pred


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='GNN baselines on pcqm4m with Pytorch Geometrics')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--gnn', type=str, default='gin-virtual',
                        help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)')
    parser.add_argument('--graph_pooling', type=str, default='sum',
                        help='graph pooling strategy mean or sum (default: sum)')
    parser.add_argument('--drop_ratio', type=float, default=0,
                        help='dropout ratio (default: 0)')
    parser.add_argument('--num_layers', type=int, default=5,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=600,
                        help='dimensionality of hidden units in GNNs (default: 600)')
    parser.add_argument('--train_subset', action='store_true')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    parser.add_argument('--log_dir', type=str, default="",
                        help='tensorboard log directory')
    parser.add_argument('--use_triplet_loss', action='store_true')
    parser.add_argument('--checkpoint_dir', type=str, default = '', help='directory to save checkpoint')
    parser.add_argument('--save_test_dir', type=str, default = '', help='directory to save test submission file')
    args = parser.parse_args()

    print(args)

    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    random.seed(42)

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    ### automatic dataloading and splitting
    dataset = PygPCQM4MDataset(root = 'dataset/')

    split_idx = dataset.get_idx_split()

    ### automatic evaluator. takes dataset name as input
    evaluator = PCQM4MEvaluator()

    if args.use_triplet_loss:
        if args.train_subset:
            subset_ratio = 0.1
            subset_idx = torch.randperm(len(split_idx["train"]))[:int(subset_ratio*len(split_idx["train"]))]
            anchor_loader = DataLoader(dataset[split_idx["train"][subset_idx]], batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
            positive_loader = DataLoader(dataset[split_idx["train"][subset_idx]], batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
            negative_loader = DataLoader(dataset[split_idx["train"][subset_idx]], batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
        else:
            anchor_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
            positive_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
            negative_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
    elif args.train_subset:
        subset_ratio = 0.1
        subset_idx = torch.randperm(len(split_idx["train"]))[:int(subset_ratio*len(split_idx["train"]))]
        train_loader = DataLoader(dataset[split_idx["train"][subset_idx]], batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
    else:
        train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)

    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)

    if args.save_test_dir is not '':
        test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)

    if args.checkpoint_dir is not '':
        os.makedirs(args.checkpoint_dir, exist_ok = True)

    shared_params = {
        'num_layers': args.num_layers,
        'emb_dim': args.emb_dim,
        'drop_ratio': args.drop_ratio,
        'graph_pooling': args.graph_pooling
    }

    if args.gnn == 'gin':
        model = GNN(gnn_type = 'gin', virtual_node = False, **shared_params).to(device)
    elif args.gnn == 'gin-virtual':
        model = GNN(gnn_type = 'gin', virtual_node = True, **shared_params).to(device)
    elif args.gnn == 'gcn':
        model = GNN(gnn_type = 'gcn', virtual_node = False, **shared_params).to(device)
    elif args.gnn == 'gcn-virtual':
        model = GNN(gnn_type = 'gcn', virtual_node = True, **shared_params).to(device)
    elif args.gnn == 'gin-virtual-bnn':
        model = BayesianGNN(gnn_type = 'gin', virtual_node = True, **shared_params).to(device)
    else:
        raise ValueError('Invalid GNN type')
       
    num_params = sum(p.numel() for p in model.parameters())
    print(f'#Params: {num_params}')

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    if args.log_dir is not '':
        writer = SummaryWriter(log_dir=args.log_dir)

    best_valid_mae = 1000

    if args.train_subset:
        scheduler = StepLR(optimizer, step_size=300, gamma=0.25)
        args.epochs = 1000
    else:
        scheduler = StepLR(optimizer, step_size=30, gamma=0.25)

    # start epoch (default = 1, unless resuming training)
    firstEpoch = 1
    # check if checkpoint exist -> load model
    checkpointFile = os.path.join(args.checkpoint_dir, 'checkpoint.pt')
    if os.path.exists(checkpointFile):
        # load checkpoint file
        print("Loading existing weights from {}".format(checkpointFile))
        checkpointData = torch.load(checkpointFile)
        firstEpoch = checkpointData["epoch"]
        model.load_state_dict(checkpointData["model_state_dict"])
        optimizer.load_state_dict(checkpointData["optimizer_state_dict"])
        scheduler.load_state_dict(checkpointData["scheduler_state_dict"])
        best_valid_mae = checkpointData["best_valid_mae"]
        num_params = checkpointData["num_params"]

    if args.use_triplet_loss:
        model.gnn_node.register_forward_hook(get_activation('gnn_node'))

    for epoch in range(firstEpoch, args.epochs + 1):
        print("=====Epoch {}".format(epoch))
        print('Training...')
        if args.use_triplet_loss:
            train_mae = triplet_loss_train(model, device, anchor_loader, negative_loader, positive_loader, optimizer, args.gnn)
        else:
            train_mae = train(model, device, train_loader, optimizer, args.gnn)

        print('Evaluating...')
        valid_mae = eval(model, device, valid_loader, evaluator)

        print({'Train': train_mae, 'Validation': valid_mae})

        if args.log_dir is not '':
            writer.add_scalar('valid/mae', valid_mae, epoch)
            writer.add_scalar('train/mae', train_mae, epoch)

        if valid_mae < best_valid_mae:
            best_valid_mae = valid_mae
            if args.checkpoint_dir is not '':
                print('Saving checkpoint...')
                checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(), 'best_val_mae': best_valid_mae, 'num_params': num_params}
                torch.save(checkpoint, os.path.join(args.checkpoint_dir, 'checkpoint.pt'))

            if args.save_test_dir is not '':
                print('Predicting on test data...')
                y_pred = test(model, device, test_loader)
                print('Saving test submission file...')
                evaluator.save_test_submission({'y_pred': y_pred}, args.save_test_dir)

        scheduler.step()
            
        print(f'Best validation MAE so far: {best_valid_mae}')

    if args.log_dir is not '':
        writer.close()


if __name__ == "__main__":
    main()
