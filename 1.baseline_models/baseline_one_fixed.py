from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve, auc


def training_with_split_one(model, data):
    f1_scores = list()
    auroc_scores = list()
    auprc_scores = list()
    ap50_scores = list()
    frequency = list()

    relations = set(data['se'])

    for i, rel in enumerate(relations):
        print(i)
        se_data = data[data['se'] == rel]
        y = se_data['label']
        x = se_data[['node1', 'node2']].to_numpy().tolist()

        # one hot encode dataset
        x = MultiLabelBinarizer().fit_transform(x)

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


# samples and negative samples of the full dataset created on r script
neg = pd.read_csv('data/negative_samples.csv')
pos = pd.read_csv('data/positive_samples.csv')

# poly se with more than 500 appearances
pop_se = pd.read_csv('se.csv')
pop_se = set(pop_se['se'])

pos = pos[pos['se'].isin(pop_se)]
neg = neg[neg['se'].isin(pop_se)]

# assign labels and concatenate
neg['label'] = 0
pos['label'] = 1
full = pd.concat([pos, neg])

# training and evaluate
lr = LogisticRegression()
f1, auroc, auprc, ap50_sc, freq = training_with_split_new(lr, full)

mean_auprc = sum(auprc) / len(auprc)
mean_auroc = sum(auroc) / len(auroc)
mean_ap50 = sum(ap50_sc) / len(ap50_sc)
mean_freq = sum(freq) / len(freq)
df = pd.DataFrame(
    {'auprc': auprc, 'auroc': auroc, 'ap50': ap50_sc, 'f1_score': f1, 'freq': freq})
df.to_csv('results/baseline_one/logistic_neg_sampling.csv')