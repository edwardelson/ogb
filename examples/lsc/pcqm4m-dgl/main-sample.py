from ogb.lsc import DglPCQM4MDataset, PCQM4MEvaluator

import argparse
import dgl
import numpy as np
import os
import random
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from gnn import GNN, DiffPoolGNN, BayesDiffPoolGNN

reg_criterion = torch.nn.L1Loss()


def collate_dgl(samples):
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    labels = torch.stack(labels)

    return batched_graph, labels


def train(model, device, loader, optimizer, gnn_name):
    model.train()
    loss_accum = 0

    for step, (bg, labels) in enumerate(tqdm(loader, desc="Iteration")):
        bg = bg.to(device)
        x = bg.ndata.pop('feat')
        edge_attr = bg.edata.pop('feat')
        labels = labels.to(device)

        pred = model(bg, x, edge_attr).view(-1,)
        optimizer.zero_grad()
        loss = reg_criterion(pred, labels)
        
        if loss =='gin-virtual-bayes-diffpool':
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

    for step, (bg, labels) in enumerate(tqdm(loader, desc="Iteration")):
        bg = bg.to(device)
        x = bg.ndata.pop('feat')
        edge_attr = bg.edata.pop('feat')
        labels = labels.to(device)

        with torch.no_grad():
            pred = model(bg, x, edge_attr).view(-1, )

        y_true.append(labels.view(pred.shape).detach().cpu())
        y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)["mae"]


def test(model, device, loader):
    model.eval()
    y_pred = []

    for step, (bg, _) in enumerate(tqdm(loader, desc="Iteration")):
        bg = bg.to(device)
        x = bg.ndata.pop('feat')
        edge_attr = bg.edata.pop('feat')

        with torch.no_grad():
            pred = model(bg, x, edge_attr).view(-1, )

        y_pred.append(pred.detach().cpu())

    y_pred = torch.cat(y_pred, dim=0)

    return y_pred

from ogb.lsc import DglPCQM4MDataset, PCQM4MEvaluator
import os.path as osp
from dgl.data.utils import load_graphs, save_graphs, Subset
from ogb.utils import smiles2graph
from ogb.utils.torch_util import replace_numpy_with_torchtensor
from ogb.utils.url import decide_download, download_url, extract_zip
class SampleDglPCQM4MDataset(DglPCQM4MDataset):
    
    @property
    def raw_file_names(self):
        return 'sample_data.csv.gz'

    def prepare_graph(self):
        processed_dir = osp.join(self.folder, 'processed')
        raw_dir = osp.join(self.folder, 'raw')
        pre_processed_file_path = osp.join(processed_dir, 'dgl_data_processed')

        if osp.exists(pre_processed_file_path):        
            # if pre-processed file already exists
            self.graphs, label_dict = load_graphs(pre_processed_file_path)
            self.labels = label_dict['labels']
        else:
            # if pre-processed file does not exist
            
            if not osp.exists(osp.join(raw_dir, 'sample_data.csv.gz')):
                # if the raw file does not exist, then download it.
                self.download()

            data_df = pd.read_csv(osp.join(raw_dir, 'sample_data.csv.gz'))
            smiles_list = data_df['smiles']
            homolumogap_list = data_df['homolumogap']

            print('Converting SMILES strings into graphs...')
            self.graphs = []
            self.labels = []
            for i in tqdm(range(len(smiles_list))):

                smiles = smiles_list[i]
                homolumogap = homolumogap_list[i]
                graph = self.smiles2graph(smiles)
                
                assert(len(graph['edge_feat']) == graph['edge_index'].shape[1])
                assert(len(graph['node_feat']) == graph['num_nodes'])

                dgl_graph = dgl.graph((graph['edge_index'][0], graph['edge_index'][1]), num_nodes = graph['num_nodes'])
                dgl_graph.edata['feat'] = torch.from_numpy(graph['edge_feat']).to(torch.int64)
                dgl_graph.ndata['feat'] = torch.from_numpy(graph['node_feat']).to(torch.int64)

                self.graphs.append(dgl_graph)
                self.labels.append(homolumogap)

            self.labels = torch.tensor(self.labels, dtype=torch.float32)

            # double-check prediction target
            split_dict = self.get_idx_split()
            assert(all([not torch.isnan(self.labels[i]) for i in split_dict['train']]))
            assert(all([not torch.isnan(self.labels[i]) for i in split_dict['valid']]))
            assert(all([torch.isnan(self.labels[i]) for i in split_dict['test']]))

            print('Saving...')
            save_graphs(pre_processed_file_path, self.graphs, labels={'labels': self.labels})
        
    
    # just modify the get_idx_split function to call our new filename
    def get_idx_split(self):
        # NOTE: CHANGED split_dict.pt to sample_split_dict.pt
        split_dict = replace_numpy_with_torchtensor(torch.load(osp.join(self.folder, 'sample_split_dict.pt')))
        return split_dict


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='GNN baselines on pcqm4m with DGL')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed to use (default: 42)')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--gnn', type=str, default='gin-virtual',
                        help='GNN to use, which can be from '
                             '[gin, gin-virtual, gcn, gcn-virtual] (default: gin-virtual)')
    parser.add_argument('--graph_pooling', type=str, default='sum',
                        help='graph pooling strategy mean or sum (default: sum)')
    parser.add_argument('--drop_ratio', type=float, default=0,
                        help='dropout ratio (default: 0)')
    parser.add_argument('--num_layers', type=int, default=5,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=600,
                        help='dimensionality of hidden units in GNNs (default: 600)')
    parser.add_argument('--train_subset', action='store_true',
                        help='use 10% of the training set for training')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    parser.add_argument('--log_dir', type=str, default="",
                        help='tensorboard log directory. If not specified, '
                             'tensorboard will not be used.')
    parser.add_argument('--checkpoint_dir', type=str, default='',
                        help='directory to save checkpoint')
    parser.add_argument('--save_test_dir', type=str, default='',
                        help='directory to save test submission file')
    args = parser.parse_args()

    print(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        device = torch.device("cuda:" + str(args.device))
    else:
        device = torch.device("cpu")

    ### automatic dataloading and splitting
    dataset = SampleDglPCQM4MDataset(root='dataset/')

    # split_idx['train'], split_idx['valid'], split_idx['test']
    # separately gives a 1D int64 tensor
    split_idx = dataset.get_idx_split()
    split_idx["train"] = split_idx["train"].type(torch.LongTensor)
    split_idx["test"] = split_idx["test"].type(torch.LongTensor)
    split_idx["valid"] = split_idx["valid"].type(torch.LongTensor)

    ### automatic evaluator.
    evaluator = PCQM4MEvaluator()

    if args.train_subset:
        subset_ratio = 0.1
        subset_idx = torch.randperm(len(split_idx["train"]))[:int(subset_ratio * len(split_idx["train"]))]
        train_loader = DataLoader(dataset[split_idx["train"][subset_idx]], batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, collate_fn=collate_dgl)
    else:
        train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, collate_fn=collate_dgl)

    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, collate_fn=collate_dgl)

    if args.save_test_dir is not '':
        test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, collate_fn=collate_dgl)

    if args.checkpoint_dir is not '':
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    shared_params = {
        'num_layers': args.num_layers,
        'emb_dim': args.emb_dim,
        'drop_ratio': args.drop_ratio,
        'graph_pooling': args.graph_pooling
    }

    if args.gnn == 'gin':
        model = GNN(gnn_type='gin', virtual_node=False, **shared_params).to(device)
    elif args.gnn == 'gin-virtual':
        model = GNN(gnn_type='gin', virtual_node=True, **shared_params).to(device)
    elif args.gnn == 'gcn':
        model = GNN(gnn_type='gcn', virtual_node=False, **shared_params).to(device)
    elif args.gnn == 'gcn-virtual':
        model = GNN(gnn_type='gcn', virtual_node=True, **shared_params).to(device)
    elif args.gnn == 'gin-virtual-diffpool':
        model = DiffPoolGNN(gnn_type='gin', virtual_node=True, **shared_params).to(device)
    elif args.gnn == 'gin-virtual-bayes-diffpool':
        model = BayesDiffPoolGNN(gnn_type='gin', virtual_node=True, **shared_params).to(device)
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

    """ load from latest checkpoint """
    # start epoch (default = 1, unless resuming training)
    firstEpoch = 1
    # check if checkpoint exist -> load model
    checkpointFile = os.path.join(args.checkpoint_dir, 'checkpoint.pt')
    if os.path.exists(checkpointFile):
        # load checkpoint file
        checkpointData = torch.load(checkpointFile)
        firstEpoch = checkpointData["epoch"]
        model.load_state_dict(checkpointData["model_state_dict"])
        optimizer.load_state_dict(checkpointData["optimizer_state_dict"])
        scheduler.load_state_dict(checkpointData["scheduler_state_dict"])
        best_valid_mae = checkpointData["best_val_mae"]
        num_params = checkpointData["num_params"]
        print("Loaded existing weights from {}. Continuing from epoch: {} with best valid MAE: {}".format(checkpointFile, firstEpoch, best_valid_mae))

        
    for epoch in range(firstEpoch, args.epochs + 1):
        print("=====Epoch {}".format(epoch))
        print('Training...')
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
                checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(),
                              'optimizer_state_dict': optimizer.state_dict(),
                              'scheduler_state_dict': scheduler.state_dict(), 'best_val_mae': best_valid_mae,
                              'num_params': num_params}
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
