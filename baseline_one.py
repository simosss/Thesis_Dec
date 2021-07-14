from sklearn.preprocessing import MultiLabelBinarizer
from collections import defaultdict
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve, auc
import pandas as pd


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
        drugs.update([stitch_id1, stitch_id2])
    fin.close()
    n_interactions = sum([len(v) for v in combo2se.values()])
    print('Drug combinations: %d Side effects: %d' % (len(combo2stitch), len(se2name)))
    print('Drug-drug interactions: %d' % n_interactions)
    return combo2stitch, combo2se, se2name, drugs


def training(model, x_tr, x_te, yy, yy_test):
    for i in range(yy.shape[1]):
        print(i)
        model.fit(x_tr, yy[:, i])
        print("finished training")
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


# poly side effects with more than 500 occurences
se_500 = pd.read_csv('se.csv')
se_500 = se_500['# poly_side_effects'].to_list()

# load polypharmacy dataset
combo2stitch, combo2se, se2name, drugs = load_combo_se()

# map each drug to integer
drugs_dict = {i: val for val, i in enumerate(sorted(drugs, reverse=False))}

# create datasets with drug pairs and their respective poly se
labels = list()
pairs = list()
for combo in sorted(combo2se.keys()):
    labels.append(list(combo2se[combo]))
    pairs.append(list(combo2stitch[combo]))

# replace drug with its id
x = list()
for pair in pairs:
    x.append([drugs_dict.get(item, item) for item in pair])

# one hot encode dataset and targets
y = MultiLabelBinarizer().fit_transform(labels)
x = MultiLabelBinarizer().fit_transform(x)

# split into train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# chose only some relations for check
#y_mini = y_train[:, 0:10]
#y_mini_test = y_test[:, 0:10]

# training and evaluate
f1_scores = list()
auroc_scores = list()
auprc_scores = list()
lr = LogisticRegression(random_state=1, max_iter=1000)
# svm = SVC(class_weight='balanced', random_state=1, probability=True, max_iter=100)

f1, auroc, auprc = training(lr, x_train, x_test, y_train, y_test)


# Frequency of the se
freq = y_test.sum(axis=0) / len(y_test)
mean_auprc = sum(auprc) / len(auprc)
mean_freq = sum(freq) / len(freq)
df = pd.DataFrame({'auprc': auprc, 'auroc': auroc, 'f1_score': f1, 'freq': freq})
df.to_csv('results/baseline_one/try_1.csv')





