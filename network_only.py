import pandas as pd
import networkx as nx
from  tqdm import tqdm
import numpy as np
import os

# path to files with facts
# current example with tab-separated triples of the KG in the form head-rel-tail
# path_to_files = "../local_rules/data/WN18RR/"
# loading the files
df_train = pd.read_csv("raw/decagon_train.csv")
df_test = pd.read_csv('raw/decagon_test.csv')

# some stats
rels = sorted(df_train.relation.unique())
unique_ents = sorted(list(set(df_train['node1'].unique().tolist() + df_train['node2'].unique().tolist() + df_test['node1'].unique().tolist() + df_test['node2'].unique().tolist())))
print(f'Unique relations: {len(rels)}')
print(f'Unique ents: {len(unique_ents)}')
print(f'# of triples in train: {len(df_train)}')
print(f'# of triples in test: {len(df_test)}')


# This will create a dict where for each relation (key) we will have a networkx graph (value)
network_dict = {}
for rel in rels:
    # Keep only triples from this relation
    print(f'Relation: {rel}')
    subset = df_train[df_train['relation'] == rel][['node1', 'node2']]
    # Create tuples of (head, tail) for input to networkx
    node_pairs = list(zip(subset['node1'].tolist(), subset['node2'].tolist()))
    # Create networkx graph
    G = nx.Graph()
    # Populate it
    G.add_edges_from(node_pairs)
    # Add nodes that do not exist already (these will be isolated)
    for node in unique_ents:
        if not(G.has_node(node)):
            G.add_node(node)
    # Save it
    network_dict[rel] = G
    # Print some stats
    print(f'{nx.info(G)}')
    print('~'*50)



# Create two networkx graphs with all train/test data (multi-relational)

train_G = nx.from_pandas_edgelist(df_train, source='node1', target='node2', edge_attr='relation',
                        create_using=nx.MultiGraph)
test_G = nx.from_pandas_edgelist(df_test, source='node1', target='node2', edge_attr='relation',
                        create_using=nx.MultiGraph)


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
y_train = []
# Two helper dictionaries that map the relation string to an integer.
rel2id = dict(zip(rels, [i for i in range(len(rels))]))
id2rels = dict(zip([i for i in range(len(rels))], rels))

# Iterating over all train triples
for head_true, tail_true, type_dic in tqdm(train_G.edges(data=True)):
    # This will hold the 44 features for each triple
    cur_feat = []
    # For each relation we will use the corresponding single-relational network
    for i, rel in enumerate(rels):
        # Append the 4 features generated from each network to the whole feature vector
        cur_feat.extend(get_preds(network_dict[rel], head_true, tail_true))
    X_train.append(cur_feat)
    y_train.append(rel2id[type_dic['relation']])


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
for head_true, tail_true, type_dic in tqdm.tqdm_notebook(test_G.edges(data=True)):
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

