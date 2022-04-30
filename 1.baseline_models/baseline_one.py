from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
import pandas as pd
from helpers import load_combo_se, training_with_split


# load polypharmacy dataset
combo2stitch, combo2se, se2name, drugs = load_combo_se()

# map each drug to integer
drugs_dict = {val: i for i, val in enumerate(sorted(drugs, reverse=False))}

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

# y_mini = y[:, :10]

# training and evaluate
bayes = LogisticRegression()
f1, auroc, auprc, ap50, freq = training_with_split(bayes, x, y)

mean_auprc = sum(auprc) / len(auprc)
mean_freq = sum(freq) / len(freq)
df = pd.DataFrame(
    {'auprc': auprc, 'auroc': auroc, 'ap50': ap50, 'f1_score': f1, 'freq': freq})
df.to_csv('results/baseline_one/logistic.csv')
