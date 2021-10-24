import pandas as pd
import networkx as nx
from  tqdm import tqdm
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split
import os

def load_combo_se(fname='data/bio-decagon-combo.csv'):
    combo2stitch = {}
    combo2se = defaultdict(set)
    se2name = {}
    drugs = set()
    fin = open(fname)
    print( 'Reading: %s' % fname)
    fin.readline()
    for line in fin:
        stitch_id1, stitch_id2, se, se_name = line.strip().split(',')
        if se not in se_500:
            continue
        combo = stitch_id1 + '_' + stitch_id2
        combo2stitch[combo] = [stitch_id1, stitch_id2]
        combo2se[combo].add(se)
        se2name[se] = se_name
        drugs.update([stitch_id1,stitch_id2])
    fin.close()
    n_interactions = sum([len(v) for v in combo2se.values()])
    print('Drug combinations: %d Side effects: %d' % (len(combo2stitch), len(se2name)))
    print('Drug-drug interactions: %d' % n_interactions)
    return combo2stitch, combo2se, se2name, drugs

# path to files with facts
# current example with tab-separated triples of the KG in the form head-rel-tail
# path_to_files = "../local_rules/data/WN18RR/"
# loading the files
#df_train = pd.read_csv("raw/decagon_train.csv")
#df_test = pd.read_csv('raw/decagon_test.csv')

se_500 = pd.read_csv('se.csv')
se_500 = se_500['# poly_side_effects'].to_list()

combo2stitch, combo2se, se2name, drugs = load_combo_se()

labels = list()
pairs = list()
for combo in sorted(combo2se.keys()):
    labels.append(list(combo2se[combo]))
    pairs.append(list(combo2stitch[combo]))

# lab = list()
# data = list()
# for drug1 in drugs:
#     for drug2 in drugs:
#         data.append([drug1, drug2])
#         if [drug1, drug2] in pairs or [drug2, drug1] in pairs:
#             lab.append(1)
#         else:
#             lab.append(0)
#
# left = [dat[0] for dat in data]
# right = [dat[1] for dat in data]
# del data
# df = pd.DataFrame({'head': left, 'tail': right, 'label': lab})
# x = df[['head', 'tail']]
# y = df[['label']]

left = [pair[0] for pair in pairs]
right = [pair[1] for pair in pairs]
lab = [1 for pair in pairs]

df = pd.DataFrame({'head': left, 'tail': right, 'label': lab})
x = df[['head', 'tail']]
y = df[['label']]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#node_pairs = list(zip(x_train['head'].tolist(), x_train['tail'].tolist()))
#test_pairs = list(zip(x_test['head'].tolist(), x_test['tail'].tolist()))
# Create networkx graph
#G = nx.Graph()
# Populate it
#G.add_edges_from(node_pairs)

train_G = nx.from_pandas_edgelist(x_train, source='head', target='tail', create_using=nx.Graph)
test_G = nx.from_pandas_edgelist(x_test, source='head', target='tail', create_using=nx.Graph)

# # This will create a dict where for each relation (key) we will have a networkx graph (value)
# network_dict = {}
# for rel in rels:
#     # Keep only triples from this relation
#     print(f'Relation: {rel}')
#     subset = df_train[df_train['relation'] == rel][['node1', 'node2']]
#     # Create tuples of (head, tail) for input to networkx
#     node_pairs = list(zip(subset['node1'].tolist(), subset['node2'].tolist()))
#     # Create networkx graph
#     G = nx.Graph()
#     # Populate it
#     G.add_edges_from(node_pairs)
#     # Add nodes that do not exist already (these will be isolated)
#     for node in unique_ents:
#         if not(G.has_node(node)):
#             G.add_node(node)
#     # Save it
#     network_dict[rel] = G
#     # Print some stats
#     print(f'{nx.info(G)}')
#     print('~'*50)



# Create two networkx graphs with all train/test data (multi-relational)

# train_G = nx.from_pandas_edgelist(df_train, source='head', target='tail', edge_attr='label',
#                         create_using=nx.MultiGraph)
# test_G = nx.from_pandas_edgelist(df_test, source='head', target='tail', edge_attr='label',
#                         create_using=nx.MultiGraph)


# CREATE TRAIN SAMPLES

# Creating train in the form of X = (head, tail) with the class Y = rel
# For each train sample we will create a feature vector with some structural scores per relation.
# The example here is with 4 structural scores (adamic_adar_index, jaccard_coefficient, resource_allocation_index, preferential_attachment) per relation.
# As such for each sample we will have 11 (relations) x 4 (structural score) = 44 features


def get_pred(graph, head, tail, func=nx.adamic_adar_index):
    """Simple wrapper to get the structural score for only one triple from the networkx function."""
    try:
        return [item for item in func(graph, [(head, tail)])][0][2]
    except ArithmeticError:
        return 0


def get_preds(cur_network, head_true, tail_true):
    """Function that given a graph and the node pair, return an array with 4 structural scores"""
    aa = get_pred(cur_network, head_true, tail_true, func=nx.adamic_adar_index)
    jc = get_pred(cur_network, head_true, tail_true, func=nx.jaccard_coefficient)
    ra = get_pred(cur_network, head_true, tail_true, func=nx.resource_allocation_index)
    pa = get_pred(cur_network, head_true, tail_true, func=nx.preferential_attachment)
    return np.array([aa, jc, ra, pa])


# This will be the training samples array with size Num_triples X 44. Each triple will have the 44 structural triples.
X_train = []
# This will contain the relation of a triple as a class.
Y_train = []

# Iterating over all train triples
for head_true, tail_true in tqdm(train_G.edges(data=True)):
    # This will hold the 44 features for each triple
    # For each relation we will use the corresponding single-relational network
    # Append the 4 features generated from each network to the whole feature vector
    feat = get_preds(train_G, head_true, tail_true)
    X_train.append(feat)
    y_train.append(1)


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# classifier = RandomForestClassifier(max_depth = 10,
#                             n_estimators=100,
#                             random_state= 42,
#                             n_jobs=-1)
# A classifier
classifier = LogisticRegression(random_state=42, solver='liblinear')
classifier.fit(np.array(X_train), y_train)


# Same procedure to create the test samples as well

X_test = []
y_test = []
for head_true, tail_true, type_dic in tqdm(test_G.edges(data=True)):
    if not(head_true in train_G and tail_true in train_G):
        continue
    cur_feat = []
    for i, rel in enumerate(rels):
        cur_feat.extend(get_preds(network_dict[rel], head_true, tail_true))
    X_test.append(cur_feat)
    y_test.append(rel2id[type_dic['rel']])



from sklearn.metrics import classification_report, confusion_matrix
preds = classifier.predict(np.array(X_test))
print(classification_report(y_test, preds))
print(confusion_matrix(y_test, preds))

