import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from difflib import SequenceMatcher
# import Levenshtein

# Levenshtein.ratio('hello world', 'hello')


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


def hist(metric, xlabel=None, ylabel=None):
    """plot a histogram of the given metric"""

    plt.hist(metric, 100, color='red')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def remap(df):
    """Changes values of df based on the mapping"""

    df['node1'] = df['node1'].map(nodes_dict)
    df['node2'] = df['node2'].map(nodes_dict)
    df['relation'] = df['relation'].map(rel_dict)
    return df


train = pd.read_csv('raw/decagon_train.csv')
val = pd.read_csv('raw/decagon_validation.csv')
test = pd.read_csv('raw/decagon_test.csv')
mono = pd.read_csv('data/bio-decagon-mono.csv', header=0, names=['drug', 'se', 'se_name'])
full_data = pd.concat([train, val, test])

extra_graphs = train[train.relation.isin(['interacts', 'targets'])]
ppi = train[train.relation.isin(['interacts'])]
target = train[train.relation.isin(['targets'])]


targeted_proteins = target['node2'].unique()
# 3648
targeting_drugs = target['node1'].unique()
# 284

first_order = ppi.node1.isin(targeted_proteins) | ppi.node2.isin(targeted_proteins)
first_order_ppi = ppi[first_order]
# 319.409

# filter out protein - protein and protein - drug edges
train = train[~train.relation.isin(['interacts', 'targets'])]
combo = pd.concat([train, val, test])

# semi-target combos 3.448.736
semi_targeting = combo.node1.isin(targeting_drugs) | combo.node2.isin(targeting_drugs)
semi_targeting_combos = combo[semi_targeting]

# full-target combos 1.188.034
targeting = combo.node1.isin(targeting_drugs) & combo.node2.isin(targeting_drugs)
targeting_combos = combo[targeting]

# frequency of each poly se
foo = combo['relation'].value_counts()
plt.hist(foo, 100, color='red')
plt.xlabel('appearances')
plt.ylabel('number of side effects')
plt.show()


foo[foo < 10000].sum()
# 2.706.315 edges
foo[foo < 5000].sum()
# 1.297.378 edges
len(foo[foo < 5000])
# 643 se

# Create dictionaries that map every node and every relation to an integer respectively
r = set(combo['relation'])
nodes = set(combo['node1']).union(set(combo['node2']))
num_nodes = len(nodes)
rel_dict = {i: val for val, i in enumerate(sorted(r, reverse=False))}
nodes_dict = {i: val for val, i in enumerate(sorted(nodes, reverse=False))}
inv_nodes_dict = {v: k for k, v in nodes_dict.items()}
del train, test, val

# Recreate the original dataset with the integer values for nodes and relations
all_d = remap(combo)

pairs = all_d[['node1', 'node2']]
pairs['comb'] = pairs['node1'].astype(str) + '_' + pairs['node2'].astype(str)
pair_freq = pairs['comb'].value_counts()
plt.hist(pair_freq, 100, color='red')
plt.xlabel('Number of pairs')
plt.ylabel('Polypharmacy side effects')
plt.show()
unique_pairs = pairs.drop_duplicates()
nod = set(pairs['node1']).union(set(pairs['node2']))


# mono analysis

mono_se = mono['se'].unique()
# 10.184
fre = mono['se'].value_counts()
bre = mono['se_name'].value_counts()
len(fre[fre > 50])
# 913
len(fre[fre > 10])
# 3.637


mono_target = mono[mono.drug.isin(targeting_drugs)]
mono_se_t = mono_target['se'].unique()
# 8.519
fre_t = mono_target['se'].value_counts()
len(fre_t[fre_t > 50])
# 263
len(fre_t[fre_t > 10])
# 2092

floo = fre_t[fre_t > 10]
hist(floo)

# possible se duplicates
floo = floo.to_frame().reset_index()
se = floo['index']
names = mono_target.se_name[mono_target.se.isin(se)].unique()

sim = list()
simil = dict()
for i, phr_1 in enumerate(names[:100]):
    for j, phr_2 in enumerate(names[:100]):
        if i > j:
            simil[phr_1 + '---' + phr_2] = similar(phr_1, phr_2)
