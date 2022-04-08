# Based on https://github.com/pyg-team/pytorch_geometric/blob/master/examples/hetero/hetero_link_pred.py
# and https://github.com/pyg-team/pytorch_geometric/issues/3958

import argparse
import os.path as osp
import sys
import traceback

import numpy as np

import torch
import torch.nn.functional as F
from torch.nn import Linear, Softmax

import torch_geometric.transforms as T
from torch_geometric.nn import GATConv, to_hetero

import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--use_weighted_loss', action='store_true',
                    help='Whether to use weighted MSE loss.')
args = parser.parse_args()

data = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('../data/data.pkl', 'rb') as pickle_file:
    data = pickle.load(pickle_file)

    data['materials']['x'] = data['materials']['x'].to(torch.float32)
    data['concepts']['x'] = data['concepts']['x'].to(torch.float32)
    data['materials', 'links', 'concepts']['edge_index'] = data['materials',
                                                                'links', 'concepts']['edge_index'].to(torch.int64)
    data['materials', 'links', 'concepts']['edge_label'] = data['materials',
                                                                'links', 'concepts']['edge_label'].to(torch.long)
    data['materials', 'links', 'concepts']['edge_label'] = data['materials',
                                                                'links', 'concepts']['edge_label'].add(0)

# path = osp.join(osp.dirname(osp.realpath(__file__)), '../../data/MovieLens')
# dataset = MovieLens(path, model_name='all-MiniLM-L6-v2')
# data = dataset[0].to(device)

# Add a reverse ('movie', 'rev_rates', 'user') relation for message passing:
data = T.ToUndirected()(data)

"""
data['materials', 'links', 'concepts']['edge_label'] = data['materials',
                                                            'links', 'concepts']['edge_label'].long()
data['concepts', 'rev_links', 'materials']['edge_label'] = data['concepts',
                                                                'rev_links', 'materials']['edge_label'].long()
"""
# del data['movie', 'rev_rates', 'user'].edge_label  # Remove "reverse" label.
# print(data['materials', 'links', 'concepts'].edge_label.shape)
# Perform a link-level split into training, validation, and test edges:
train_data, val_data, test_data = T.RandomLinkSplit(
    num_val=0.1,
    num_test=0.1,
    neg_sampling_ratio=1.0,
    add_negative_train_samples=True,
    is_undirected=True,
    edge_types=[('materials', 'links', 'concepts')],
    rev_edge_types=[('concepts', 'rev_links', 'materials')],
)(data)

# print(train_data['materials', 'links', 'concepts'].edge_label.shape,
#      train_data['materials', 'links', 'concepts'].edge_label)

# TODO: Use pageRank as weight for the loss of each edge?
if args.use_weighted_loss:
    weight = torch.bincount(
        train_data['materials', 'links', 'concepts'].edge_label)
    weight = weight.max() / weight
else:
    weight = None

"""
def weighted_mse_loss(pred, target, weight=None):
    weight = 1. if weight is None else weight[target].to(pred.dtype)
    return (weight * (pred - target.to(pred.dtype)).pow(2)).mean()
"""


loss_function = torch.nn.CrossEntropyLoss()


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
        # 2 * hidden_channels because we have applied to_hetero
        # with the parameter aggr='sum' which has led to making
        # a 2 * hidden_channels tensor
        self.lin1 = Linear(2 * hidden_channels, hidden_channels)

        # Return a tensor with 2 outputs, one for existing edges
        # the other for non-existing edges
        self.lin2 = Linear(hidden_channels, 2)

        # Return a softmax of z so we can
        # use it as a probability distribution
        self.softmax = Softmax(dim=-1)

    def forward(self, z_dict, edge_label_index):
        row, col = edge_label_index
        z = torch.cat([z_dict['materials'][row],
                       z_dict['concepts'][col]], dim=-1)

        z = self.lin1(z).relu()
        z = self.lin2(z)

        # Return the softmax of the output
        return self.softmax(z)


class Model(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.encoder = GNNEncoder(hidden_channels, hidden_channels)
        self.encoder = to_hetero(self.encoder, data.metadata(), aggr='sum')
        self.decoder = EdgeDecoder(hidden_channels)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        z_dict = self.encoder(x_dict, edge_index_dict)
        return self.decoder(z_dict, edge_label_index)


model = Model(hidden_channels=128).to(device)

# Due to lazy initialization, we need to run one model step so the number
# of parameters can be inferred:


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

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train():
    model.train()
    optimizer.zero_grad()

    # Get outputs from the model
    pred = model(train_data.x_dict, train_data.edge_index_dict,
                 train_data['materials', 'links', 'concepts'].edge_label_index)  # train_data['materials', 'concepts'].edge_label_index

    # Predictions by the model are tensors where the sum of probabilities = 1
    # we just get the argmax from those tensors for each prediction
    # pred = torch.argmax(pred, dim=1)

    # We get the real value we should be predicting (1=exists, 0=does not exist)
    target = train_data['materials', 'links',
                        'concepts'].edge_label.to(torch.long)
    target = torch.nn.functional.one_hot(target).to(torch.float)

    # Calculate the loss
    loss = loss_function(pred, target)

    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test(data):
    model.eval()
    pred = model(data.x_dict, data.edge_index_dict,
                 data['materials', 'links', 'concepts'].edge_label_index)
    pred = pred.clamp(min=0, max=5)
    target = data['materials', 'links', 'concepts'].edge_label.long()
    target = torch.nn.functional.one_hot(target).to(torch.float)

    # Calculate the loss
    loss = loss_function(pred, target)

    # loss.backward()
    return float(loss)


"""
@torch.no_grad()
def test(data):
    model.eval()
    pred = model(data.x_dict, data.edge_index_dict,
                 data['materials', 'links', 'concepts'].edge_label_index)
    pred = pred.clamp(min=0, max=5)
    target = data['materials', 'links', 'concepts'].edge_label.long()
    target = torch.nn.functional.one_hot(target).to(torch.float)
    rmse = F.mse_loss(pred, target).sqrt()
    return float(rmse)
"""

EPOCHS = 300
current_epoch = 0
train_rmse = None
val_rmse = None
test_rmse = None
loss = None

for epoch in range(1, EPOCHS):
    loss = train()
    train_rmse = test(train_data)
    val_rmse = test(val_data)
    test_rmse = test(test_data)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_rmse:.4f}, '
          f'Val: {val_rmse:.4f}, Test: {test_rmse:.4f}')
    current_epoch = epoch

torch.save({
    'epoch': current_epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}, './model.pt')

"""
activation = {}


def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


model.decoder.lin1.register_forward_hook(get_activation('lin1'))
output = model(train_data)
activation['fc3']
"""
