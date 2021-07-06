from collections import defaultdict
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
import sys

# check the size of the variables
# local_vars = list(locals().items())
# for var, obj in local_vars:
#     print(var, sys.getsizeof(obj))


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


def load_mono_se(fname='data/bio-decagon-mono.csv'):
    stitch2se = defaultdict(set)
    se2name = {}
    fin = open(fname)
    print('Reading: %s' % fname)
    fin.readline()
    for line in fin:
        contents = line.strip().split(',')
        stitch_id, se, = contents[:2]
        se_name = ','.join(contents[2:])
        stitch2se[stitch_id].add(se)
        se2name[se] = se_name
    return stitch2se, se2name

se_500 = pd.read_csv('se.csv')
se_500 = se_500['# poly_side_effects'].to_list()



combo2stitch, combo2se, se2name, drugs = load_combo_se()
stitch2se, se2name_mono = load_mono_se()

mono_se_dict = {i: val for val, i in enumerate(sorted(se2name_mono.keys(), reverse=False))}

# create lists with pairs and se of each pair
labels = list()
pairs = list()
for combo in sorted(combo2se.keys()):
    labels.append(list(combo2se[combo]))
    pairs.append(list(combo2stitch[combo]))

# one-hot-encode the target
y = MultiLabelBinarizer().fit_transform(labels)
y_sparse = sparse.csr_matrix(y)
del labels, combo2stitch, combo2se, se2name_mono

# transform the dataset

x = list()
for pair in pairs:
    x.append([stitch2se.get(item, item) for item in pair])

left = [list(x[i][0]) for i in range(len(x))]
right = [list(x[i][1]) for i in range(len(x))]
del x, pairs, pair

l = list()
for lef in left:
    l.append([str(mono_se_dict.get(item, item)) for item in lef])
del left, lef
l.insert(0, [str(i) for i in range(len(mono_se_dict))])
# remove first element from the list after the ohe is done correctly
ll = MultiLabelBinarizer().fit_transform(l)
ll = ll[1:, :10184]
del l

r = list()
for rig in right:
    r.append([str(mono_se_dict.get(item, item)) for item in rig])
del right, rig
r.insert(0, [str(i) for i in range(len(mono_se_dict))])
rr = MultiLabelBinarizer().fit_transform(r)
rr = rr[1:, :10184]
del r

x = ll + rr
del ll, rr

np.savetxt('foo.csv', x, delimiter=',', fmt='%1.0f')
np.savetxt('bar.csv', y, delimiter=',', fmt='%1.0f')

x_sparse = sparse.csr_matrix(x)

del x

svd = TruncatedSVD(n_components=300, random_state=42)
lsa_x_300 = svd.fit_transform(x_sparse)
print(svd.explained_variance_ratio_.sum())

#pca = PCA(n_components=0.99)
#x_pca = pca.fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(lsa_x, y_sparse, test_size=0.2, random_state=42)
del x, y, lsa_x
y_mini = y_train[:, :]
y_mini_test = y_test[:, :]

# sklearn accepts only array-like as targets, not sparse
y_dense = y_mini.toarray()
y_dense_test = y_mini_test.toarray()


# Plan A: USe sklearn implementation. Too much time. Only one score accross all se
lr = LogisticRegression(random_state=1, max_iter=800)
multi_target_lr = MultiOutputClassifier(lr, n_jobs=-1)
multi_target_lr.fit(x_train, y_dense)
# y_pred = multi_target_lr.predict(x_test)
score = multi_target_lr.score(x_test, y_mini_test[:, 0:10])


# Plan B: One relation at a time
#accuracies = list()
#f1_scores = list()
auc_scores = list()
lr = LogisticRegression(random_state=1, max_iter=1000)
for i in range(y_dense.shape[1]):
    print(i)
    lr.fit(x_train, y_dense[:, i])
    # acc = lr.score(x_test, y_dense_test[:, i])
    # y_pred = lr.predict(x_test)
    y_prob = lr.predict_proba(x_test)
    # f1 = f1_score(y_dense_test[:, i], y_pred)
    auc = roc_auc_score(y_dense_test[:, i], y_prob[:, 1])
    auc_scores.append(auc)
    # accuracies.append(acc)
    # f1_scores.append(f1)
    # print(acc)
    # print(f1)

# Frequency of the se
freq = y.sum(axis=0)
