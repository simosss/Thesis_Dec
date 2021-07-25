from sklearn.preprocessing import MultiLabelBinarizer
# from sklearn.multioutput import MultiOutputClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from helpers import load_combo_se, training_with_split


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

y_mini = y[:, 10]

# training and evaluate
f1_scores = list()
auroc_scores = list()
auprc_scores = list()
# lr = LogisticRegression(random_state=1, max_iter=1000)
bayes = GaussianNB()

f1, auroc, auprc, freq = training_with_split(bayes, x, y_mini)

mean_auprc = sum(auprc) / len(auprc)
mean_freq = sum(freq) / len(freq)
df = pd.DataFrame({'auprc': auprc, 'auroc': auroc, 'f1_score': f1, 'freq': freq})
df.to_csv('results/baseline_one/try_bayes.csv')
