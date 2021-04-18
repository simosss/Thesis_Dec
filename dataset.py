import os.path as osp

import numpy as np
import pandas as pd
from collections import defaultdict

import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data


class Dec(InMemoryDataset):

    def __init__(self, root='', name='Decagon', transform=None, pre_transform=None):
        self.name = name
        self.root = root
        super(Dec, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, 'processed')

    @property
    def num_relations(self):
        return self.data.edge_type.max().item() + 1

    @property
    def raw_file_names(self):
        # return ['decagon_train.csv', 'decagon_validation.csv', 'decagon_test.csv', 'bio-decagon-mono.csv']
        return ['sample.csv', 'val_sample.csv', 'test_sample.csv', 'bio-decagon-mono.csv']

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        pass

    def process(self):

        def create_edge_index_edge_type(df):
            """Given a df with rows holding triplets n1, n2, relation
             creates edge_index and edge_type tensors in the form PyG wants it"""

            ei = df[["node1", "node2"]].to_numpy().transpose()
            et = df["relation"].to_numpy().transpose()
            e_index = torch.from_numpy(ei)
            e_type = torch.from_numpy(et)
            # e_index = torch.tensor(ei, dtype=torch.long).contiguous()
            # e_type = torch.tensor(et, dtype=torch.long).contiguous()
            return e_index, e_type

        def remap(df):
            """Changes values of df based on the mapping"""

            df['node1'] = df['node1'].map(nodes_dict)
            df['node2'] = df['node2'].map(nodes_dict)
            df['relation'] = df['relation'].map(rel_dict)
            return df

        # read  datasets
        train = pd.read_csv(self.raw_paths[0])
        val = pd.read_csv(self.raw_paths[1])
        test = pd.read_csv(self.raw_paths[2])
        mono = pd.read_csv(self.raw_paths[3])

        # Create dictionaries that map every node and every relation to an integer respectively
        n1 = set(train['node1'])
        n2 = set(train['node2'])
        r = set(train['relation'])

        nodes = n1.union(n2)
        num_nodes = len(nodes)
        # num_relations = len(r)
        rel_dict = {i: val for val, i in enumerate(sorted(r, reverse=True))}
        nodes_dict = {i: val for val, i in enumerate(sorted(nodes, reverse=True))}
        inv_nodes_dict = {v: k for k, v in nodes_dict.items()}

        # Recreate the original dataset with the integer values for nodes and relations
        train = remap(train)
        val = remap(val)
        test = remap(test)

        # create a dictionary to link mono side effects to their names
        mono_sename_dict = {}
        for (se, se_name) in zip(mono['Individual Side Effect'], mono['Side Effect Name']):
            mono_sename_dict[se] = se_name
        num_features = len(mono_sename_dict)

        # create a dictionary to link drugs to their mono se
        drug_se_dict = defaultdict(set)
        for (drug, se) in zip(mono['STITCH'], mono['Individual Side Effect']):
            drug_se_dict[drug].add(se)

        # a list of all mono side effects
        side_effects = sorted(list((mono_sename_dict.keys())))

        # create a dict of nparrays holding the feature vectors for every drug (which has mono se)
        drug_features = {}
        for drug in drug_se_dict:
            vector = np.zeros(len(side_effects))
            mono_se_found_indexes = [side_effects.index(mono_se) for mono_se in drug_se_dict[drug]]
            vector[mono_se_found_indexes] = 1
            drug_features[drug] = vector

        # A dictionary that for every node holds its feature vector. (Based on mono se if applicable or else, random)
        features = {}
        for node in nodes:
            if node in drug_se_dict:
                features[node] = drug_features[node]
            else:
                features[node] = np.random.randint(2, size=num_features)

        # Does what the previous two loops combined. not obviously faster

        # features = {}
        # for node in nodes:
        #     if node in drug_se_dict:
        #         vector = np.zeros(num_features)
        #         mono_se_found_indexes = [side_effects.index(mono_se) for mono_se in drug_se_dict[node]]
        #         vector[mono_se_found_indexes] = 1
        #         features[node] = list(vector)
        #     else:
        #         features[node] = list(np.random.randint(2, size=num_features))

        print("Finished with creation of dictionaries")

        # Create the tensor that holds the features for all the nodes

        x_list = [features[inv_nodes_dict[i]] for i in range(num_nodes)]
        x = torch.tensor(x_list, dtype=torch.float)
        print("Finished creating features")

        # Create edge index, edge type
        edge_index, edge_type = create_edge_index_edge_type(train)
        val_edge_index, val_edge_type = create_edge_index_edge_type(val)
        test_edge_index, test_edge_type = create_edge_index_edge_type(test)

        data = Data(edge_index=edge_index,
                    x=x,
                    edge_type=edge_type,
                    val_edge_index=val_edge_index,
                    val_edge_type=val_edge_type,
                    test_edge_index=test_edge_index,
                    test_edge_type=test_edge_type,
                    num_nodes=num_nodes)

        # # x = torch.rand(data.num_nodes, num_features, dtype=torch.float)
        # # data.nodes=nodes

        data, slices = self.collate([data])
        torch.save((data, slices), self.processed_paths[0])

    def __repr__(self):
        return '{}{}()'.format(self.name.upper(), self.__class__.__name__)
