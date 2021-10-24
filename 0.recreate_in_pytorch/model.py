import torch
import torch.nn as nn
from torch_geometric.nn import GAE, GCNConv, RGCNConv
import numpy as np
import torch.nn.functional as F


def init_glorot(in_channels, out_channels, dtype=torch.float32):
    """Create a weight variable with Glorot & Bengio (AISTATS 2010)
    initialization.
    """
    init_range = np.sqrt(6.0 / (in_channels + out_channels))
    initial = -init_range + 2 * init_range * \
        torch.rand((in_channels, out_channels), dtype=dtype)
    initial = initial.requires_grad_(True)
    return initial


class Encoder(nn.Module):
    def __init__(self, input_dim=10184, hid_dim=64, out_dim=32, num_relations=964, num_bases=2):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.num_relations = num_relations
        self.num_bases = num_bases
        self.conv1 = RGCNConv(self.input_dim, self.hid_dim, self.num_relations, self.num_bases)
        self.conv2 = RGCNConv(self.hid_dim, self.out_dim, self.num_relations, self.num_bases)

    def forward(self, x, edge_index, edge_type):
        x = self.conv1(x, edge_index, edge_type)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_type)
        return x


class DistMultDecoder(nn.Module):

    def __init__(self, num_relations=964, input_dim=32, dropout=0.,
                 activation=torch.sigmoid):
        super().__init__()
        self.input_dim = input_dim
        self.num_relations = num_relations
        self.dropout = dropout
        self.activation = activation

        self.local_variation = [
            torch.flatten(init_glorot(input_dim, 1))
            for _ in range(num_relations)]

    def forward(self, x, edge_index, edge_type):
        # We give a mini batch of the edge index along with the edge type of each edge
        # We seperate the two columns of edge index
        left, right = edge_index
        out = torch.empty(0, dtype=torch.double)
        # for each relation type
        for i in range(self.num_relations):
            # we keep the relative part of the edge_index (the edges of this particular type)
            sub_left = left[torch.where(edge_type == i)]
            sub_right = right[torch.where(edge_type == i)]
            # the corresponding x matrices
            inputs_left = F.dropout(x[sub_left], 0)
            inputs_right = F.dropout(x[sub_right], 0)
            # the matrix of the particular relation
            relation = torch.diag(self.local_variation[i])

            product1 = torch.mm(inputs_left, relation)
            product2 = (product1 * inputs_right).sum(dim=1)
            out = torch.cat((out, self.activation(product2)))

        return out


class DEDICOMDecoder(nn.Module):
    """DEDICOM Tensor Factorization Decoder model layer for link prediction."""

    def __init__(self, num_relations=964, input_dim=32, dropout=0.,
                 activation=torch.sigmoid):
        super().__init__()
        self.input_dim = input_dim
        self.num_relations = num_relations
        self.dropout = dropout
        self.activation = activation

        self.global_interaction = init_glorot(input_dim, input_dim)

        self.local_variation = [
            torch.flatten(init_glorot(input_dim, 1))
            for _ in range(num_relations)]

    def forward(self, x, edge_index, edge_type):
        # We give a mini batch of the edge index along with the edge type of eache edge
        # We seperate the two columns of edge index
        left, right = edge_index
        out = torch.empty(0, dtype=torch.double)
        # for each relation type
        for i in range(self.num_relations):
            # we keep the relative part of the edge_index (the edges of this particular type)
            sub_left = left[torch.where(edge_type == i)]
            sub_right = right[torch.where(edge_type == i)]
            # the corresponding x matrices
            inputs_left = F.dropout(x[sub_left], 0)
            inputs_right = F.dropout(x[sub_right], 0)
            # the matrix of the particular relation
            relation = torch.diag(self.local_variation[i])

            product1 = torch.mm(inputs_left, relation)
            product2 = torch.mm(product1, self.global_interaction)
            product3 = torch.mm(product2, relation)
            # if we were doing the calculation for a single edge, product4 would be a scalar
            # now with minibatch it is n x n
            # where n the size of minibatch (actually, size of minibatch with the particuar relation)
            # so we only care about the diagonal
            # In the original implementaton I don't see this .diag step or sth similar
            # Maybe there is a more efficient way. I hope diag() is not calculated after every single
            # coputation takes place. Otherwise it is completely waste of time
            # product4 = torch.mm(product3, inputs_right.transpose(0, 1))
            # tested it and works just tthe same
            product4_simply = (product3 * inputs_right).sum(dim=1)
            out = torch.cat((out, self.activation(product4_simply)))

        return out


# class Net(GAE):
#     def __init__(self):
#         super(Net, self).__init__()
