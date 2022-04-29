import os.path as osp
import argparse
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.datasets import Planetoid, Coauthor, Amazon
from torch_geometric.utils import train_test_split_edges
from torch_geometric.nn import GAE, VGAE, APPNP, GCNConv
import torch_geometric.transforms as T


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='VGNAE')
parser.add_argument('--dataset', type=str, default='Cora')
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--channels', type=int, default=128)
parser.add_argument('--hidden_channels', type=int, default=128)
parser.add_argument('--scaling_factor', type=float, default=1.8)
parser.add_argument('--training_rate', type=float, default=0.8)
parser.add_argument('--learning_rate', type=float, default=0.005)
args = parser.parse_args()

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', args.dataset)

if args.dataset in ['Cora', 'CiteSeer', 'PubMed']:
    dataset = Planetoid(path, args.dataset, 'public')
elif args.dataset in ['cs', 'physics']:
    dataset = Coauthor(path, args.dataset)
elif args.dataset in ['computers', 'photo']:
    dataset = Amazon(path, args.dataset)
else:
    raise ValueError


data = dataset.data
data = T.NormalizeFeatures()(data)


class Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, edge_index):
        super(Encoder, self).__init__()

        if args.model in ['GNAE', 'VGNAE']:
            self.linear1 = nn.Linear(in_channels, out_channels)
            self.linear2 = nn.Linear(in_channels, out_channels)
            self.propagate = APPNP(K=1, alpha=0)
        else:
            self.c11 = GCNConv(in_channels, args.hidden_channels)
            self.b11 = nn.BatchNorm1d(args.hidden_channels)

            self.c12 = GCNConv(args.hidden_channels, out_channels)
            self.c22 = GCNConv(args.hidden_channels, out_channels)

    def forward(self, x, edge_index, not_prop=0):
        if args.model == 'GNAE':
            x = self.linear1(x)
            x = F.normalize(x, p=2, dim=1) * args.scaling_factor
            x = self.propagate(x, edge_index)
            return x

        if args.model == 'GAE':
            x = self.b11(self.c11(x, edge_index)).relu()
            x = self.c12(x, edge_index)
            return x

        if args.model == 'VGNAE':
            xm = self.linear1(x)
            xm = F.normalize(xm, p=2, dim=1) * args.scaling_factor
            xm = self.propagate(xm, edge_index)

            x_ = self.linear2(x)
            x_ = self.propagate(x_, edge_index)
            return xm, x_

        if args.model == 'VGAE':
            x = self.c11(x, edge_index)
            x = self.b11(x).relu()

            xm = self.c12(x, edge_index)

            x_ = self.c22(x, edge_index)
            return xm, x_

        return x


class Encoder2(torch.nn.Module):
    def __init__(self, in_channels, out_channels, edge_index, dropout_rate=0.4):
        super(Encoder2, self).__init__()
        assert args.model == 'MLP'

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, args.hidden_channels),
            nn.BatchNorm1d(args.hidden_channels),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(args.hidden_channels, out_channels),
        )

    def forward(self, x, edge_index, not_prop=0):
        return self.mlp(x)


dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
channels = args.channels

train_rate = args.training_rate
val_ratio = (1-args.training_rate) / 3
test_ratio = (1-args.training_rate) * 2 / 3

# Ratio as mentioned in paper
# val_ratio = (1 - train_rate) / 4
# test_ratio = (1 - train_rate) * 3 / 4

data = train_test_split_edges(data.to(dev), val_ratio=val_ratio, test_ratio=test_ratio)

N = int(data.x.size()[0])
if args.model in ['GNAE', 'GAE']:
    model = GAE(Encoder(data.x.size()[1], channels, data.train_pos_edge_index)).to(dev)

if args.model in ['MLP']:
    model = GAE(Encoder2(data.x.size()[1], channels, data.train_pos_edge_index)).to(dev)

if args.model in ['VGNAE', 'VGAE']:
    model = VGAE(Encoder(data.x.size()[1], channels, data.train_pos_edge_index)).to(dev)

data.train_mask = data.val_mask = data.test_mask = data.y = None
x, train_pos_edge_index = data.x.to(dev), data.train_pos_edge_index.to(dev)
optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1.e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)


def train():
    model.train()
    optimizer.zero_grad()
    z  = model.encode(x, train_pos_edge_index)
    loss = model.recon_loss(z, train_pos_edge_index)
    if args.model in ['VGAE', 'VGNAE']:
        loss = loss + (1 / data.num_nodes) * model.kl_loss()
    loss.backward()
    optimizer.step()
    scheduler.step()
    return loss


def test(pos_edge_index, neg_edge_index, plot_his=0, test_model=None):
    if test_model is not None:
        model_ = test_model
    else:
        model_ = model

    model_.eval()
    with torch.no_grad():
        z = model_.encode(x, train_pos_edge_index)
        return model_.test(z, pos_edge_index, neg_edge_index)


best_auc = None
best_ap = None
best_auc_model = None
best_ap_model = None


for epoch in range(1, args.epochs):
    loss = train()
    loss = float(loss)
    
    with torch.no_grad():
        val_pos, val_neg = data.val_pos_edge_index, data.val_neg_edge_index
        auc, ap = test(data.val_pos_edge_index, data.val_neg_edge_index)

        if best_auc is None or auc > best_auc:
            best_auc = auc
            best_auc_model = deepcopy(model)

        if best_ap is None or ap > best_ap:
            best_ap = ap
            best_ap_model = deepcopy(model)

        print('Epoch: {:03d}, TRAIN_LOSS: {:.4f}, VAL_AUC: {:.4f}, VAL_AP: {:.4f}'.format(epoch, loss, auc, ap))

    # lr_scheduler.step()

print('*' * 50)
print('Training complete!')
print('*' * 50)

print('Best VAL_AUC:', best_auc)
print('Best VAL_AP:', best_ap)
print('*' * 50)

if best_auc_model is not None:
    print('Testing best AUC model!')
    with torch.no_grad():
        test_pos, test_neg = data.test_pos_edge_index, data.test_neg_edge_index
        auc, ap = test(data.test_pos_edge_index, data.test_neg_edge_index, test_model=best_auc_model)
        print(f'TEST_AUC: {auc:.5f}, TEST_AP: {ap:.5f}')

if best_ap_model is not None:
    print('Testing best AP model!')
    with torch.no_grad():
        test_pos, test_neg = data.test_pos_edge_index, data.test_neg_edge_index
        auc, ap = test(data.test_pos_edge_index, data.test_neg_edge_index, test_model=best_ap_model)
        print(f'TEST_AUC: {auc:.5f}, TEST_AP: {ap:.5f}')
