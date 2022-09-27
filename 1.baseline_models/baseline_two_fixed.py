from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
import pandas as pd
from helpers import training_with_split, load_mono_se
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve, auc
from sklearn.decomposition import PCA
import random

num_features = 300


def ap50(prob, y_true):
    prob = prob.tolist()
    y_true = y_true.tolist()
    tup = list(zip(prob, y_true))
    # sorts by the biggest probability first
    res = sorted(tup, key=lambda x: x[0], reverse=True)
    res = res[:50]
    # get only the y_true again
    label = [x[1] for x in res]
    true_positives = 0
    summ = 0
    for i, lab in enumerate(label):
        if lab == 1:
            true_positives += 1
        a = (true_positives * lab) / (i+1)
        summ += a
    # each side effect has more than 50 occurences in test set
    # so it is safe to divide by 50
    avg_p = summ / len(label)
    return avg_p


def training_with_split_two(model, data, vectors):
    f1_scores = list()
    auroc_scores = list()
    auprc_scores = list()
    ap50_scores = list()
    frequency = list()

    relations = set(data['se'])

    for idx, rel in enumerate(relations):
        print(idx)
        se_data = data[data['se'] == rel]
        y = se_data['label']

        # create x dataset
        x = se_data[['node1', 'node2']]
        x.loc[:, 'head'] = x['node1'].map(vectors)
        x.loc[:, 'tail'] = x['node2'].map(vectors)
        x.loc[:, 'pair'] = x['head'] + x['tail']
        x = pd.DataFrame(x["pair"].to_list())

        # split into train and test
        x_tr, x_te, y_tr, y_te = train_test_split(
            x, y, test_size=0.2, random_state=42, stratify=y)

        model.fit(x_tr, y_tr)
        y_pred = model.predict(x_te)
        y_prob = model.predict_proba(x_te)
        # keep probability for the positive class only
        y_prob = y_prob[:, 1]
        f1_s = f1_score(y_te, y_pred)
        auroc_s = roc_auc_score(y_te, y_prob)
        precision, recall, thresholds = precision_recall_curve(y_te, y_prob)
        auprc_s = auc(recall, precision)
        ap50_s = ap50(y_prob, y_te)
        f1_scores.append(f1_s)
        auroc_scores.append(auroc_s)
        auprc_scores.append(auprc_s)
        ap50_scores.append(ap50_s)
        frequency.append(y_te.sum() / len(y_te))

    return f1_scores, auroc_scores, auprc_scores, ap50_scores, frequency


def training_with_split_two_v2(model, positive, negative, vectors):
    f1_scores = list()
    auroc_scores = list()
    auprc_scores = list()
    ap50_scores = list()
    frequency = list()

    relations = set(positive['se'])

    for idx, rel in enumerate(relations):
        print(idx)
        se_pos = positive[positive['se'] == rel]
        se_neg = negative[negative['se'] == rel]

        # split into train and test
        num_samples = int(0.8 * len(se_pos))
        train_index = random.sample(list(se_pos.index), num_samples)

        x_tr_p = se_pos.loc[train_index]
        x_tr_n = se_neg.loc[train_index]
        x_tr = pd.concat([x_tr_p, x_tr_n])
        x_tr = x_tr.sample(frac=1)
        y_tr = x_tr['label']

        x_te_p = se_pos.drop(train_index, axis=0)
        x_te_n = se_neg.drop(train_index, axis=0)
        x_te = pd.concat([x_te_p, x_te_n])
        x_te = x_te.sample(frac=1)
        y_te = x_te['label']

        # create x dataset
        x_tr = x_tr[['node1', 'node2']]
        x_tr.loc[:, 'head'] = x_tr['node1'].map(vectors)
        x_tr.loc[:, 'tail'] = x_tr['node2'].map(vectors)
        x_tr.loc[:, 'pair'] = x_tr['head'] + x_tr['tail']
        x_tr = pd.DataFrame(x_tr["pair"].to_list())

        x_te = x_te[['node1', 'node2']]
        x_te.loc[:, 'head'] = x_te['node1'].map(vectors)
        x_te.loc[:, 'tail'] = x_te['node2'].map(vectors)
        x_te.loc[:, 'pair'] = x_te['head'] + x_te['tail']
        x_te = pd.DataFrame(x_te["pair"].to_list())

        model.fit(x_tr, y_tr)
        y_pred = model.predict(x_te)
        y_prob = model.predict_proba(x_te)
        # keep probability for the positive class only
        y_prob = y_prob[:, 1]
        f1_s = f1_score(y_te, y_pred)
        auroc_s = roc_auc_score(y_te, y_prob)
        precision, recall, thresholds = precision_recall_curve(y_te, y_prob)
        auprc_s = auc(recall, precision)
        ap50_s = ap50(y_prob, y_te)
        f1_scores.append(f1_s)
        auroc_scores.append(auroc_s)
        auprc_scores.append(auprc_s)
        ap50_scores.append(ap50_s)
        frequency.append(y_te.sum() / len(y_te))

    return f1_scores, auroc_scores, auprc_scores, ap50_scores, frequency


# CREATE FEATURES
stitch2se, se2name_mono = load_mono_se()

drugs = list()
mono_se = list()
for key, val in stitch2se.items():
    drugs.append(key)
    mono_se.append(val)

mlb = MultiLabelBinarizer()
features = mlb.fit_transform(mono_se)

pca = PCA(n_components=num_features)
features = pca.fit_transform(features)
print(sum(pca.explained_variance_ratio_))

mapping = dict()
for i, drug in enumerate(drugs):
    mapping[drug] = list(features[i, :num_features])

# CREATE DATASET
# samples and negative samples of the full dataset created on r script
neg = pd.read_csv('data/negative_samples.csv')
pos = pd.read_csv('data/positive_samples.csv')
# poly se with more than 500 appearances
pop_se = pd.read_csv('se.csv')
pop_se = set(pop_se['se'])
pos = pos[pos['se'].isin(pop_se)]
neg = neg[neg['se'].isin(pop_se)]
# drugs with no mono se
pos = pos[pos['node1'].isin(drugs)]
neg = neg[neg['node1'].isin(drugs)]
pos = pos[pos['node2'].isin(drugs)]
neg = neg[neg['node2'].isin(drugs)]
# make sure that every triplet has its negative and vise versa, otherwise get rid of it
common = pos.index.intersection(neg.index)
pos = pos.loc[common]
neg = neg.loc[common]
# assign labels and concatenate
neg['label'] = 0
pos['label'] = 1
# full = pd.concat([pos, neg])

# sample1 = full.head(100000)
# sample2 = full.tail(100000)
# sample = pd.concat([sample1, sample2])

lr = LogisticRegression(random_state=1)

# training -----------------------------------------------
f1, auroc, auprc, ap50, freq = training_with_split_two_v2(lr, pos, neg, mapping)

# Export
mean_auprc = sum(auprc) / len(auprc)
mean_auroc = sum(auroc) / len(auroc)
mean_ap50 = sum(ap50) / len(ap50)
mean_freq = sum(freq) / len(freq)
df = pd.DataFrame({'auprc': auprc, 'auroc': auroc, 'f1_score': f1, 'ap50': ap50, 'freq': freq})
df.to_csv('results/baseline_two/logistic_neg_sampling_v2.csv')
