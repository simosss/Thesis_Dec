# import numpy as np
# import pandas as pd
from sklearn.metrics import roc_auc_score
import torch
from torch_geometric.nn import GAE
import torch.nn.functional as F
# import torch.optim as optim
# import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils
from torch_geometric.data import NeighborSampler
# ------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------

from model import Encoder, DEDICOMDecoder
from dataset import Dec

dataset = Dec()
data = dataset[0]

data.train_pos_edge_index = data.edge_index
data.val_pos_edge_index = data.edge_index
data.test_pos_edge_index = data.edge_index


# ------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------

# returns a negative index with exactly one negative edge for each positive
def neg_sampling(edge_index, num_nodes):
    struc_neg_sampl = pyg_utils.structured_negative_sampling(edge_index, num_nodes)
    i, j, k = struc_neg_sampl
    i = i.tolist()
    k = k.tolist()
    neg_edge_index = [i, k]
    neg_edge_index = torch.tensor(neg_edge_index)
    return neg_edge_index


# returns a tensor of labels 1 or 0
def get_link_labels(pos_edge_index, neg_edge_index):
    conc = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(conc, dtype=torch.float)
    link_labels[:pos_edge_index.size(1)] = 1.
    return link_labels


# ------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------
def train():
    model.train()

    for batch_size, n_id, adjs in train_loader:
        pos_edge_index, ind, _ = adjs[0]
        neg_edge_index = neg_sampling(pos_edge_index, len(n_id))
        pos_edge_type = data.edge_type[ind]
        neg_edge_type = data.edge_type[ind]
        z = model.encode(data.x[n_id], pos_edge_index, pos_edge_type)
        # see where this needs to be
        optimizer.zero_grad()
        # total_edge_index = torch.cat((pos_edge_index, neg_edge_index), 1)
        # total_edge_type = torch.cat((pos_edge_type, neg_edge_type), 0)
        pos_link_logits = model.decode(z, pos_edge_index, pos_edge_type)
        neg_link_logits = model.decode(z, neg_edge_index, neg_edge_type)
        # link_labels = get_link_labels(pos_edge_index, neg_edge_index)
        loss = F.binary_cross_entropy_with_logits(pos_link_logits, torch.ones(pos_edge_index.size(1))) + \
            F.binary_cross_entropy_with_logits(neg_link_logits, torch.zeros(pos_edge_index.size(1)))
        loss.backward()
        optimizer.step()

    return loss


@torch.no_grad()
def test():
    model.eval()
    perfs = []
    z, nid = model.inference(data.x)  # remember to fix this for actual validation set
    pos_edge_index = data.val_pos_edge_index
    neg_edge_index = neg_sampling(pos_edge_index,
                                  len(data.edge_type))  # remember to fix this edge_type len for validatio set
    link_logits = model.decode(z, pos_edge_index, neg_edge_index)
    link_probs = link_logits.sigmoid()
    link_labels = get_link_labels(pos_edge_index, neg_edge_index)
    perfs.append(roc_auc_score(link_labels, link_probs))
    return perfs


# --------------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------------
train_loader = NeighborSampler(data.train_pos_edge_index, batch_size=8, shuffle=True, sizes=[5, 5])
# subgraph_loader = NeighborSampler(data.edge_index, node_idx=None, sizes=[-1], batch_size=128, shuffle=False)

enc = Encoder()
decod = DEDICOMDecoder()
model = GAE(enc, decod)
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

# ----------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------
best_val_perf = test_perf = 0
for epoch in range(1, 3):
    train_loss = train()
    print(epoch, train_loss)
    # val_perf = test()
    # log = 'Epoch: {:03d}, Loss: {:.4f}, Val: {:.4f}'
    # print(log.format(epoch, train_loss, val_perf[0]))
