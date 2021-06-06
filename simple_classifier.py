import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.decomposition import PCA


def remap(df):
    """Changes values of df based on the mapping"""

    df['node1'] = df['node1'].map(nodes_dict)
    df['node2'] = df['node2'].map(nodes_dict)
    df['relation'] = df['relation'].map(rel_dict)
    return df


# read  datasets
train = pd.read_csv('raw/decagon_train.csv')
val = pd.read_csv('raw/decagon_validation.csv')
test = pd.read_csv('raw/decagon_test.csv')
mono = pd.read_csv('raw/bio-decagon-mono.csv')

# filter out protein - protein and protein - drug edges
train = train[~train.relation.isin(['interacts', 'targets'])]

all = pd.concat([train, val, test])


# Create dictionaries that map every node and every relation to an integer respectively
r = set(all['relation'])
nodes = set(all['node1']).union(set(all['node2']))
num_nodes = len(nodes)
rel_dict = {i: val for val, i in enumerate(sorted(r, reverse=False))}
nodes_dict = {i: val for val, i in enumerate(sorted(nodes, reverse=False))}
inv_nodes_dict = {v: k for k, v in nodes_dict.items()}

# Recreate the original dataset with the integer values for nodes and relations
train = remap(train)
val = remap(val)
test = remap(test)
mono['STITCH'] = mono['STITCH'].map(nodes_dict)

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

# Create the tensor that holds the features for all the nodes
x_list = [drug_features[drug] for drug in sorted(drug_features.keys())]
x = np.array(x_list)

pca = PCA(n_components=500)
x_pca = pca.fit_transform(x)

test['vec1'] = test['node1'].map(drug_features)
test['vec2'] = test['node2'].map(drug_features)
test = test.dropna()

# freezing
df3 = pd.DataFrame(test['vec1'].to_list())