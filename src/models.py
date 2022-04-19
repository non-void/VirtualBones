import copy
import math
import torch
import torch.nn as nn
from torch.nn import Embedding
import numpy as np
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.nn import EdgeConv, global_max_pool
from torch_geometric.utils import dropout_adj, get_laplacian


def MLP(dimensions, dropout=False, batch_norm=False, batch_norm_momentum=1e-3):
    return nn.Sequential(*[
        nn.Sequential(
            nn.Dropout(p=0.5) if dropout else nn.Identity(),
            nn.Linear(dimensions[i - 1], dimensions[i]),
            nn.PReLU(dimensions[i]),
            nn.BatchNorm1d(dimensions[i], affine=True, momentum=batch_norm_momentum) if batch_norm else nn.Identity())
        for i in range(1, len(dimensions))])


class MLPModel(nn.Module):
    def __init__(self, dim):
        super(MLPModel, self).__init__()
        self.dim = dim
        self.layers = nn.Sequential(MLP(self.dim[0:-1], batch_norm=True),
                                    nn.Linear(self.dim[-2], self.dim[-1]))

    def forward(self, x):
        return self.layers(x)


class GRU_Model(nn.Module):
    def __init__(self, input_num, hidden_num, output_num, shortcut=False):
        super(GRU_Model, self).__init__()
        self.hidden_num = hidden_num
        self.cell = nn.GRUCell(input_num, hidden_num)
        self.output_num = copy.deepcopy(output_num)
        self.output_num.insert(0, hidden_num)
        self.out_linear = nn.Sequential(MLP(self.output_num[0:-1], batch_norm=True, dropout=True),
                                        nn.Dropout(p=0),
                                        nn.Linear(self.output_num[-2], self.output_num[-1]))
        self.shortcut = shortcut

    def forward(self, x, hidden=None):
        if hidden is None:
            hidden = torch.zeros(x.shape[0], self.hidden_num).to(self.cell.bias_hh.device)
            # hidden = torch.randn(x.shape[0], self.hidden_num).to(self.cell.bias_hh.device)
        next_hidden = self.cell(x, hidden)
        y = self.out_linear(next_hidden)
        if self.shortcut:
            y = y + x[:, :-3]
            # y = y * 0 + x[:, :-3]
        return y, next_hidden


class GRU_GNN_Model(nn.Module):
    # the implementation of pytorch geometric heavily relies on scatter_sum which can not be deterministic
    # so the GNN using this package can not achieve deterministic
    def __init__(self, input_num, hidden_num, output_num, gru_out_dim, gnn_dim, edge_index, batch_norm=True):
        super(GRU_GNN_Model, self).__init__()
        self.hidden_num = hidden_num
        self.cell = nn.GRUCell(input_num, hidden_num)
        self.output_num = copy.deepcopy(output_num)
        self.output_num.insert(0, hidden_num)
        self.gnn_dim = gnn_dim
        self.gru_out_linear = nn.Sequential(MLP(self.output_num, batch_norm=batch_norm, dropout=True))
        # self.gru_out_linear = nn.Sequential(MLP(self.output_num, batch_norm=False, dropout=True))
        self.gru_out_dim = gru_out_dim
        self.out_linear = nn.Sequential(nn.Linear(self.gnn_dim[-1] + self.gru_out_dim, 3))
        self.edge_index = edge_index
        self.edge_conv_layers = nn.ModuleList()

        for i in range(len(self.gnn_dim) - 1):
            self.edge_conv_layers.append(EdgeConv(MLP([2 * self.gnn_dim[i], self.gnn_dim[i + 1]])))

    def dropout_edge(self, input_edge_index, p=0.8, force_undirected=True):
        if self.training:
            edge_index, _ = dropout_adj(input_edge_index, p=p, force_undirected=force_undirected)
        else:
            edge_index = input_edge_index
        return edge_index

    def forward(self, x, smoothed_vert_pos, hidden=None):
        batch_size = x.shape[0]
        smoothed_vert_pos = smoothed_vert_pos.view((batch_size, -1, 3))
        smoothed_x0 = torch.zeros((batch_size, smoothed_vert_pos.shape[1], self.gnn_dim[1])).to(x.device)
        smoothed_x1 = torch.zeros((batch_size, smoothed_vert_pos.shape[1], self.gnn_dim[2])).to(x.device)

        for i in range(batch_size):
            edge_index = self.dropout_edge(self.edge_index)
            smoothed_x0[i] = self.edge_conv_layers[0](smoothed_vert_pos[i], edge_index)

        if not self.training:
            smoothed_x0 = smoothed_x0.detach()

        for i in range(batch_size):
            edge_index = self.dropout_edge(self.edge_index)
            smoothed_x1[i] = self.edge_conv_layers[1](smoothed_x0[i], edge_index)

        if not self.training:
            smoothed_x1 = smoothed_x1.detach()

        if hidden is None:
            hidden = torch.zeros(x.shape[0], self.hidden_num).to(self.cell.bias_hh.device)
        next_hidden = self.cell(x, hidden)
        gru_out = self.gru_out_linear(next_hidden).view((batch_size, -1, self.gru_out_dim))
        y = self.out_linear(torch.cat([gru_out, smoothed_x1], axis=2)).view((batch_size, -1))
        return y, next_hidden


if __name__ == "__main__":
    pass
