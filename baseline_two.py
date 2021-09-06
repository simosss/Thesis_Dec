from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
import pandas as pd
# from sklearn.naive_bayes import GaussianNB
# from sklearn.tree import DecisionTreeClassifier
from sklearn.experimental import enable_hist_gradient_boosting
# from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from helpers import training_with_split, load_combo_se, load_mono_se
from scipy.sparse import hstack
from sklearn.naive_bayes import GaussianNB
import numpy as np


# load data ------------------------------------
combo2stitch, combo2se, se2name, drugs = load_combo_se()
stitch2se, se2name_mono = load_mono_se()

mono_se_dict = {val: i for i, val in enumerate(sorted(se2name_mono.keys(), reverse=False))}

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

r = list()
for rig in right:
    r.append([str(mono_se_dict.get(item, item)) for item in rig])
del right, rig
# r.insert(0, [str(i) for i in range(len(mono_se_dict))])
rr = mlb.transform(r)
rr = rr[:, :10184]
del r

x = np.concatenate((ll, rr), axis=1)
del ll, rr

# export dense dataset. Too big
# np.savetxt('foo.csv', x, delimiter=',', fmt='%1.0f')
# np.savetxt('bar.csv', y, delimiter=',', fmt='%1.0f')

x_sparse = sparse.csr_matrix(x)

del x

# PCA ------------------------------------------------
svd = TruncatedSVD(n_components=500, random_state=42)
lsa_x = svd.fit_transform(x_sparse)
print(svd.explained_variance_ratio_.sum())

# pca = PCA(n_components=0.99)
# x_pca = pca.fit_transform(x)

# prepare training ------------------------------------------
# x_train, x_test, y_train, y_test = train_test_split(lsa_x, y, test_size=0.2, random_state=42)
# del lsa_x
# y_mini = y_train[:, :5]
# y_mini_test = y_test[:, :5]

lr = LogisticRegression(random_state=1, max_iter=1000)
# tree = DecisionTreeClassifier()
boost = HistGradientBoostingClassifier()
bayes = GaussianNB()

# training -----------------------------------------------
# f1, auroc, auprc = training(lr, x_train, x_test, y_train, y_test)
f1, auroc, auprc, freq = training_with_split(lr, lsa_x, y[:, :10])

mean_auprc = sum(auprc) / len(auprc)
mean_freq = sum(freq) / len(freq)
df = pd.DataFrame({'auprc': auprc, 'auroc': auroc, 'f1_score': f1, 'freq': freq})
df.to_csv('results/baseline_two/pca300_bayes.csv')
