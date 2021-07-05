from sklearn.preprocessing import MultiLabelBinarizer
from collections import defaultdict
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score


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


combo2stitch, combo2se, se2name, drugs = load_combo_se()

drugs_dict = {i: val for val, i in enumerate(sorted(drugs, reverse=False))}

labels = list()
pairs = list()
for combo in sorted(combo2se.keys()):
    labels.append(list(combo2se[combo]))
    pairs.append(list(combo2stitch[combo]))

x = list()
for pair in pairs:
    x.append([drugs_dict.get(item, item) for item in pair])

y = MultiLabelBinarizer().fit_transform(labels)
x = MultiLabelBinarizer().fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
#y_mini = y_train[:, 0:2]
#y_mini_test = y_test[:, 0:2]

# Plan B: One relation at a time
#accuracies = list()
#f1_scores = list()
auc_scores = list()
lr = LogisticRegression(random_state=1, max_iter=1000)
for i in range(y_train.shape[1]):
    print(i)
    lr.fit(x_train, y_train[:, i])
    # acc = lr.score(x_test, y_dense_test[:, i])
    # y_pred = lr.predict(x_test)
    y_prob = lr.predict_proba(x_test)
    # f1 = f1_score(y_dense_test[:, i], y_pred)
    auc = roc_auc_score(y_test[:, i], y_prob[:, 1])
    auc_scores.append(auc)
    # accuracies.append(acc)
    # f1_scores.append(f1)
    # print(acc)
    # print(f1)

# lr = LogisticRegression(random_state=1, max_iter=400)
# multi_target_lr = MultiOutputClassifier(lr, n_jobs=-1)
# multi_target_lr.fit(x_train, y_mini)
# y_pred = multi_target_lr.predict(x_test)
# score = multi_target_lr.score(x_test, y_mini_test[:, 0:2])

