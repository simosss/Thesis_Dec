from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
import pandas as pd
import numpy as np
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from helpers import training_with_split, load_combo_se, load_mono_se, load_targets
from scipy.sparse import hstack


# load data ------------------------------------
combo2stitch, combo2se, se2name, drugs = load_combo_se()
stitch2se, se2name_mono = load_mono_se()
stitch2protein = load_targets()

# dict that maps mono se to int
mono_se_dict = {val: i for i, val in enumerate(sorted(se2name_mono.keys(), reverse=False))}

# create dict that maps all proteins that at least on drug targets to int
targets = list()
for val in stitch2protein.values():
    targets.extend(val)
targets = list(set(targets))
targets.sort()
target_dict = {val: i for i, val in enumerate(sorted(targets, reverse=False))}

# create lists with pairs and poly se of each pair ---------------------
labels = list()
pairs = list()
for combo in sorted(combo2se.keys()):
    labels.append(list(combo2se[combo]))
    pairs.append(list(combo2stitch[combo]))

# create mono se features -------------------------
x = list()
for pair in pairs:
    x.append([stitch2se.get(item, item) for item in pair])

# create separately the feature vectors of the two drugs
# and then instead of concatenate we add them

left = [list(x[i][0]) for i in range(len(x))]
right = [list(x[i][1]) for i in range(len(x))]
del x

mlb = MultiLabelBinarizer()
l = list()
for lef in left:
    l.append([str(mono_se_dict.get(item, item)) for item in lef])
del left, lef
# to be sure that every one of the 10184 mono se appears at least once
# so that the hot encoding is done properly
l.insert(0, [str(i) for i in range(len(mono_se_dict))])

ll = mlb.fit_transform(l)
# remove first element from the list after the ohe is done correctly
ll = ll[1:, :10184]
del l
ll_sparse = sparse.csr_matrix(ll)
del ll

# repeat for the second drug
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


# PCA ------------------------------------------------
svd = TruncatedSVD(n_components=100, random_state=42)
lsa_ll = svd.fit_transform(ll_sparse)
lsa_rr = svd.transform(rr_sparse)
print(svd.explained_variance_ratio_.sum())

lsa_x = np.concatenate((lsa_ll, lsa_rr), axis=1)
del lsa_ll, lsa_rr

# protein features -----------------------------

# follow the same procedure to create the target proreins vector
x_prot = list()
for pair in pairs:
    x_prot.append([stitch2protein.get(item, {}) for item in pair])

left_prot = [list(x_prot[i][0]) for i in range(len(x_prot))]
right_prot = [list(x_prot[i][1]) for i in range(len(x_prot))]
del x_prot

mlb_prot = MultiLabelBinarizer()
l_prot = list()
for lef in left_prot:
    l_prot.append([str(target_dict.get(item, item)) for item in lef])
del left_prot, lef
l_prot.insert(0, [str(i) for i in range(len(target_dict))])
# remove first element from the list after the ohe is done correctly
ll_prot = mlb_prot.fit_transform(l_prot)
ll_prot = ll_prot[1:, :len(target_dict)]
del l_prot
ll_prot_sparse = sparse.csr_matrix(ll_prot)
del ll_prot

r_prot = list()
for rig in right_prot:
    r_prot.append([str(target_dict.get(item, item)) for item in rig])
del right_prot, rig
r_prot.insert(0, [str(i) for i in range(len(target_dict))])
rr_prot = mlb_prot.transform(r_prot)
rr_prot = rr_prot[1:, :len(target_dict)]
del r_prot
rr_prot_sparse = sparse.csr_matrix(rr_prot)
del rr_prot

# PCA ------------------------------------------------
svd_prot = TruncatedSVD(n_components=50, random_state=42)
lsa_ll_prot = svd_prot.fit_transform(ll_prot_sparse)
lsa_rr_prot = svd_prot.transform(rr_prot_sparse)
print(svd_prot.explained_variance_ratio_.sum())

lsa_x_prot = np.concatenate((lsa_ll_prot, lsa_rr_prot), axis=1)

x = np.concatenate((lsa_x, lsa_x_prot), axis=1)

# one-hot-encode the target
mlb_y = MultiLabelBinarizer()
y = mlb_y.fit_transform(labels)


# prepare training ------------------------------------------

boost = HistGradientBoostingClassifier()
bayes = GaussianNB()
lr = LogisticRegression(max_iter=1000)

f1, auroc, auprc, freq = training_with_split(boost, x, y[:, :1])


# training -----------------------------------------------

mean_auprc = sum(auprc) / len(auprc)
mean_freq = sum(freq) / len(freq)

df = pd.DataFrame({'auprc': auprc, 'auroc': auroc, 'f1_score': f1, 'freq': freq})
df.to_csv('results/baseline_three/pca300_50_try_1.csv')
