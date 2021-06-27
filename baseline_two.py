from collections import defaultdict
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


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


combo2stitch, combo2se, se2name, drugs = load_combo_se()
stitch2se, se2name_mono = load_mono_se()

# drugs_dict = {i: val for val, i in enumerate(sorted(drugs, reverse=False))}
mono_se_dict = {i: val for val, i in enumerate(sorted(se2name_mono.keys(), reverse=False))}

labels = list()
pairs = list()
for combo in sorted(combo2se.keys()):
    labels.append(list(combo2se[combo]))
    pairs.append(list(combo2stitch[combo]))

y = MultiLabelBinarizer().fit_transform(labels)
del labels, combo2stitch, combo2se, se2name_mono

x = list()
for pair in pairs:
    x.append([stitch2se.get(item, item) for item in pair])


left = [list(x[i][0]) for i in range(len(x))]
right = [list(x[i][1]) for i in range(len(x))]
del x, pairs, pair

l = list()
r = list()
for lef in left:
    l.append([str(mono_se_dict.get(item, item)) for item in lef])
del left, lef
for rig in right:
    r.append([str(mono_se_dict.get(item, item)) for item in rig])
del right, rig
l.insert(0, [str(i) for i in range(len(mono_se_dict))])
r.insert(0, [str(i) for i in range(len(mono_se_dict))])

# remove first element from the list after the ohe is done correctly
ll = MultiLabelBinarizer().fit_transform(l)
ll = ll[1:, :]
del l
rr = MultiLabelBinarizer().fit_transform(r)
rr = rr[1:, :]
del r

x = np.add(ll, rr)
del ll, rr


pca = PCA(n_components=0.99)
x = pca.fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
del x, y
y_mini = y_train[:, 0:2]
y_mini_test = y_test[:, 0:2]


lr = LogisticRegression(random_state=1, max_iter=400)
multi_target_lr = MultiOutputClassifier(lr, n_jobs=-1)
multi_target_lr.fit(x_train, y_mini)
y_pred = multi_target_lr.predict(x_test)
score = multi_target_lr.score(x_test, y_mini_test[:, 0:2])