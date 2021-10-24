import pandas as pd
import numpy as np

def load_ppi(fname='data/bio-decagon-ppi.csv'):
    fin = open(fname)
    print('Reading: %s' % fname)
    fin.readline()
    edges = []
    for line in fin:
        gene_id1, gene_id2 = line.strip().split(',')
        edges += [[gene_id1, gene_id2]]
    nodes = set([u for e in edges for u in e])
    print('Edges: %d' % len(edges))
    print('Nodes: %d' % len(nodes))

    net = nx.Graph()
    net.add_edges_from(edges)
    net.remove_nodes_from(nx.isolates(net))
    net.remove_edges_from(nx.selfloop_edges(net))
    node2idx = {node: i for i, node in enumerate(net.nodes())}
    return net, node2idx


def load_targets(fname='data/bio-decagon-targets.csv'):
    stitch2proteins = defaultdict(set)
    fin = open(fname)
    print ('Reading: %s' % fname)
    fin.readline()
    for line in fin:
        stitch_id, gene = line.strip().split(',')
        stitch2proteins[stitch_id].add(gene)
    return stitch2proteins

combo2stitch, combo2se, se2name, drugs = load_combo_se()
net, node2idx = load_ppi()

stitch2proteins = load_targets(fname='bio-decagon-targets-all.csv')


def remap(df):
    """Changes values of df based on the mapping"""

    df['node1'] = df['node1'].map(nodes_dict)
    df['node2'] = df['node2'].map(nodes_dict)
    df['relation'] = df['relation'].map(rel_dict)
    return df


def ohe(df, n_nodes):
    for i in range(n_nodes):
        df.insert(i, 'col' + str(i), 0)
    print('finished inserting')
    for i in range(n_nodes):
        df.loc[df['node1'] == i, ['col' + str(i)]] = 1
        df.loc[df['node2'] == i, ['col' + str(i)]] = 1
        if i % 50 == 0:
            print(i)


train = pd.read_csv('raw/decagon_train.csv')
val = pd.read_csv('raw/decagon_validation.csv')
test = pd.read_csv('raw/decagon_test.csv')

# filter out protein - protein and protein - drug edges
train = train[~train.relation.isin(['interacts', 'targets'])]
all_d = pd.concat([train, val, test])

# Create dictionaries that map every node and every relation to an integer respectively
r = set(all_d['relation'])
nodes = set(all_d['node1']).union(set(all_d['node2']))
num_nodes = len(nodes)
rel_dict = {i: val for val, i in enumerate(sorted(r, reverse=False))}
nodes_dict = {i: val for val, i in enumerate(sorted(nodes, reverse=False))}
inv_nodes_dict = {v: k for k, v in nodes_dict.items()}
del all_d

# Recreate the original dataset with the integer values for nodes and relations
train = remap(train)
val = remap(val)
test = remap(test)

datasets = [train, test, val]
#for dataset in datasets:
#    print('working on ', len(dataset))
#    ohe(dataset, num_nodes)

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC


# y = testing2[['relation']]
# X = testing2.drop(['relation', 'node1', 'node2'], axis=1)
# x_test = testing1.drop(['relation', 'node1', 'node2'], axis=1)
# y_test = testing1[['relation']]
# clf = OneVsRestClassifier(SVC()).fit(X, y)

# testing3 =testing2[['node1', 'node2']]
# testing3 = testing3.drop_duplicates()
# y_exp = clf.predict(x_test)

# it works fine
from sklearn.preprocessing import MultiLabelBinarizer
trii = [[j for j in range(200)] for i in range(100000)]
triiiii = MultiLabelBinarizer().fit_transform(trii)

from collections import defaultdict
def load_combo_se(fname='data/bio-decagon-combo.csv'):
    combo2stitch = {}
    combo2se = defaultdict(set)
    se2name = {}
    drugs=set()
    fin = open(fname)
    print( 'Reading: %s' % fname)
    fin.readline()
    for line in fin:
        stitch_id1, stitch_id2, se, se_name = line.strip().split(',')
        combo = stitch_id1 + '_' + stitch_id2
        combo2stitch[combo] = [stitch_id1, stitch_id2]
        combo2se[combo].add(se)
        se2name[se] = se_name
        drugs.update([stitch_id1,stitch_id2])
    fin.close()
    n_interactions = sum([len(v) for v in combo2se.values()])
    print ('Drug combinations: %d Side effects: %d' % (len(combo2stitch), len(se2name)))
    print ('Drug-drug interactions: %d' % (n_interactions))
    return combo2stitch, combo2se, se2name,drugs

combo2stitch, combo2se, se2name, drugs = load_combo_se()
a = list()
c = list()
for combo in sorted(combo2se.keys()):
    a.append(list(combo2se[combo]))
    c.append(list(combo2stitch[combo]))
y = MultiLabelBinarizer().fit_transform(a)
n = list()
for l in c:
    n.append([nodes_dict.get(item, item) for item in l])

x = MultiLabelBinarizer().fit_transform(n)

from sklearn.multioutput import MultiOutputClassifier
from sklearn.tree import DecisionTreeClassifier


forest = DecisionTreeClassifier(random_state=1)
multi_target_forest = MultiOutputClassifier(forest, n_jobs=-1)
multi_target_forest.fit(x, y)