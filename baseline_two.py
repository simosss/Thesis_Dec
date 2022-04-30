from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
import pandas as pd
from helpers import training_with_split, load_combo_se, load_mono_se
import numpy as np


# load data ------------------------------------
combo2stitch, combo2se, se2name, drugs = load_combo_se()
stitch2se, se2name_mono = load_mono_se()

mono_se_dict = {
    val: i for i, val in enumerate(sorted(se2name_mono.keys(), reverse=False))}

# create lists with pairs and se of each pair ---------------------
labels = list()
pairs = list()
for combo in sorted(combo2se.keys()):
    labels.append(list(combo2se[combo]))
    pairs.append(list(combo2stitch[combo]))

# one-hot-encode the target
mlb_y = MultiLabelBinarizer()
y = mlb_y.fit_transform(labels)
# y_sparse = sparse.csr_matrix(y)
del labels, combo2stitch, combo2se, se2name_mono

# transform the dataset ------------------------------------
x = list()
for pair in pairs:
    x.append([stitch2se.get(item, item) for item in pair])

left = [list(x[i][0]) for i in range(len(x))]
right = [list(x[i][1]) for i in range(len(x))]
del x, pairs, pair

mlb = MultiLabelBinarizer()

l = list()
for lef in left:
    l.append([str(mono_se_dict.get(item, item)) for item in lef])
del left, lef
# to be sure that every one of the 10184 mono se appears at least one
# so that the ohe is done properly
l.insert(0, [str(i) for i in range(len(mono_se_dict))])
# remove first element from the list after the ohe is done correctly
ll = mlb.fit_transform(l)
# are you sure it is the last three you need to discard?
ll = ll[1:, :10184]
del l
ll_sparse = sparse.csr_matrix(ll)
del ll
svd = TruncatedSVD(n_components=300, random_state=42)
lsa_ll = svd.fit_transform(ll_sparse)
print(svd.explained_variance_ratio_.sum())

r = list()
for rig in right:
    r.append([str(mono_se_dict.get(item, item)) for item in rig])
del right, rig
# r.insert(0, [str(i) for i in range(len(mono_se_dict))])
rr = mlb.transform(r)
rr = rr[:, :10184]
del r
rr_sparse = sparse.csr_matrix(rr)
del rr
lsa_rr = svd.transform(rr_sparse)
print(svd.explained_variance_ratio_.sum())

x = np.concatenate((lsa_ll, lsa_rr), axis=1)

lr = LogisticRegression(random_state=1)
# this is much better than when I tried with less
# subsample or less estimators but it is extremelly slow
gradient_boosting = GradientBoostingClassifier(
    random_state=1, n_estimators=100, subsample=0.5, max_features='auto')

# training -----------------------------------------------
f1, auroc, auprc, ap50, freq = training_with_split(gradient_boosting, x, y[:, :10])

df = pd.DataFrame(
    {'auprc': auprc, 'auroc': auroc, 'f1_score': f1, 'ap50': ap50, 'freq': freq})
df.to_csv('results/baseline_two/gbtree_try3.csv')
