from collections import defaultdict
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve, auc
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import GradientBoostingClassifier


def load_combo_se(fname='data/bio-decagon-combo.csv'):
    combo2stitch = {}
    combo2se = defaultdict(set)
    se2name = {}
    drugs = set()
    fin = open(fname)
    print('Reading: %s' % fname)
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


def training(model, x_tr, x_te, yy, yy_test):
    for i in range(yy.shape[1]):
        print(i)
        model.fit(x_tr, yy[:, i])
        # print("finished training")
        y_pred = model.predict(x_te)
        y_prob = model.predict_proba(x_te)
        # keep probability for the positive class only
        y_prob = y_prob[:, 1]
        f1_s = f1_score(yy_test[:, i], y_pred)
        auroc_s = roc_auc_score(yy_test[:, i], y_prob)
        precision, recall, thresholds = precision_recall_curve(yy_test[:, i], y_prob)
        auprc_s = auc(recall, precision)
        f1_scores.append(f1_s)
        auroc_scores.append(auroc_s)
        auprc_scores.append(auprc_s)
    return f1_scores, auroc_scores, auprc_scores


def training_with_split(model, X, Y):
    for i in range(Y.shape[1]):
        print(i)
        # split into train and test
        x_tr, x_te, y_tr, y_te = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y[:, i])
        model.fit(x_tr, y_tr[:, i])
        print("finished training")
        y_pred = model.predict(x_te)
        y_prob = model.predict_proba(x_te)
        # keep probability for the positive class only
        y_prob = y_prob[:, 1]
        f1_s = f1_score(y_te[:, i], y_pred)
        auroc_s = roc_auc_score(y_te[:, i], y_prob)
        precision, recall, thresholds = precision_recall_curve(y_te[:, i], y_prob)
        auprc_s = auc(recall, precision)
        f1_scores.append(f1_s)
        auroc_scores.append(auroc_s)
        auprc_scores.append(auprc_s)
        frequency = y_te.sum(axis=0) / len(y_te)

    return f1_scores, auroc_scores, auprc_scores, frequency


# load data ------------------------------------
se_500 = pd.read_csv('se.csv')
se_500 = se_500['# poly_side_effects'].to_list()

combo2stitch, combo2se, se2name, drugs = load_combo_se()
stitch2se, se2name_mono = load_mono_se()

mono_se_dict = {i: val for val, i in enumerate(sorted(se2name_mono.keys(), reverse=False))}

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
ll = ll[1:, :10184]
del l

r = list()
for rig in right:
    r.append([str(mono_se_dict.get(item, item)) for item in rig])
del right, rig
# r.insert(0, [str(i) for i in range(len(mono_se_dict))])
rr = mlb.transform(r)
rr = rr[1:, :10184]
del r

x = ll + rr
del ll, rr

# export dense dataset. Too big
# np.savetxt('foo.csv', x, delimiter=',', fmt='%1.0f')
# np.savetxt('bar.csv', y, delimiter=',', fmt='%1.0f')

x_sparse = sparse.csr_matrix(x)

del x

# PCA ------------------------------------------------
svd = TruncatedSVD(n_components=300, random_state=42)
lsa_x = svd.fit_transform(x_sparse)
print(svd.explained_variance_ratio_.sum())

# pca = PCA(n_components=0.99)
# x_pca = pca.fit_transform(x)

# prepare training ------------------------------------------
# x_train, x_test, y_train, y_test = train_test_split(lsa_x, y, test_size=0.2, random_state=42)
# del lsa_x
# y_mini = y_train[:, :5]
# y_mini_test = y_test[:, :5]

#lr = LogisticRegression(random_state=1, max_iter=1000)
#tree = DecisionTreeClassifier()
boost = HistGradientBoostingClassifier()

# training -----------------------------------------------
f1_scores = list()
auroc_scores = list()
auprc_scores = list()
#f1, auroc, auprc = training(lr, x_train, x_test, y_train, y_test)
f1, auroc, auprc, freq = training_with_split(boost, lsa_x, y[:, :10])
# Frequency of the se
#freq = y_test.sum(axis=0) / len(y_test)
mean_auprc = sum(auprc) / len(auprc)
mean_freq = sum(freq) / len(freq)
df = pd.DataFrame({'auprc': auprc, 'auroc': auroc, 'f1_score': f1, 'freq': freq})
df.to_csv('results/baseline_two/pca300_bayes.csv')


