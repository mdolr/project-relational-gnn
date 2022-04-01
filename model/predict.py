
import argparse
import os.path as osp
import sys
import traceback

import torch
import torch.nn.functional as F
from torch.nn import Linear

import torch_geometric.transforms as T
from torch_geometric.nn import GATConv, to_hetero

import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--use_weighted_loss', action='store_true',
                    help='Whether to use weighted MSE loss.')
args = parser.parse_args()


class GNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GATConv((-1, -1), hidden_channels)
        self.conv2 = GATConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x


class EdgeDecoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin1 = Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 2)

    def forward(self, z_dict, edge_label_index):
        row, col = edge_label_index
        z = torch.cat([z_dict['materials'][row],
                      z_dict['concepts'][col]], dim=-1)
        z = self.lin1(z)
        z = self.lin2(z)
        return z.view(-1)


class Model(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.encoder = GNNEncoder(hidden_channels, hidden_channels)
        self.encoder = to_hetero(self.encoder, data.metadata(), aggr='sum')
        self.decoder = EdgeDecoder(hidden_channels)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        z_dict = self.encoder(x_dict, edge_index_dict)
        return self.decoder(z_dict, edge_label_index)


data = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('../data/data.pkl', 'rb') as data_file:
    data = pickle.load(data_file)

    data['materials']['x'] = data['materials']['x'].to(torch.float32)
    data['concepts']['x'] = data['concepts']['x'].to(torch.float32)
    data['materials', 'links', 'concepts']['edge_index'] = data['materials',
                                                                'links', 'concepts']['edge_index'].to(torch.int64)
    data['materials', 'links', 'concepts']['edge_label'] = data['materials',
                                                                'links', 'concepts']['edge_label'].to(torch.long)
    data['materials', 'links', 'concepts']['edge_label'] = data['materials',
                                                                'links', 'concepts']['edge_label'].add(0)
    data = T.ToUndirected()(data)

    train_data, val_data, test_data = T.RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        neg_sampling_ratio=1.0,
        add_negative_train_samples=True,
        is_undirected=True,
        edge_types=[('materials', 'links', 'concepts')],
        rev_edge_types=[('concepts', 'rev_links', 'materials')],
    )(data)

if args.use_weighted_loss:
    weight = torch.bincount(
        train_data['materials', 'links', 'concepts'].edge_label)
    weight = weight.max() / weight
else:
    weight = None


def weighted_mse_loss(pred, target, weight=None):
    weight = 1. if weight is None else weight[target].to(pred.dtype)
    return (weight * (pred - target.to(pred.dtype)).pow(2)).mean()


model = None

model = Model(hidden_channels=48).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

with torch.no_grad():
    try:
        model.encoder(train_data.x_dict, train_data.edge_index_dict)
    except AssertionError:
        _, _, tb = sys.exc_info()
        traceback.print_tb(tb)  # Fixed format
        tb_info = traceback.extract_tb(tb)
        filename, line, func, text = tb_info[-1]

        print('An error occurred on line {} in statement {}'.format(line, text))
        exit(1)

checkpoint = torch.load('./model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

material = data['materials']['x'][0]
concept = data['concepts']['x'][0]

print(train_data.x_dict)
print('-------')
print(train_data.edge_index_dict)

pred = model(train_data.x_dict, train_data.edge_index_dict,
             train_data['materials', 'links', 'concepts'].edge_label_index)
pred = pred.clamp(min=0, max=5)

print(pred)
