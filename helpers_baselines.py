from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve, auc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import MultiLabelBinarizer


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


def load_combo_se(fname='data/bio-decagon-combo.csv'):
    combo2stitch = {}
    combo2se = defaultdict(set)
    se2name = {}
    drugs = set()
    # poly side effects with more than 500 occurences
    se_500 = pd.read_csv('se.csv')
    se_500 = se_500['# poly_side_effects'].to_list()
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


def load_targets(fname='data/bio-decagon-targets.csv'):
    stitch2proteins = defaultdict(set)
    fin = open(fname)
    print('Reading: %s' % fname)
    fin.readline()
    for line in fin:
        stitch_id, gene = line.strip().split(',')
        stitch2proteins[stitch_id].add(gene)
    return stitch2proteins


def training(model, x_tr, x_te, yy, yy_test):
    f1_scores = list()
    auroc_scores = list()
    auprc_scores = list()
    ap50_scores = list()
    frequency = list()

    for i in range(yy.shape[1]):
        print(i)
        model.fit(x_tr, yy[:, i])
        y_pred = model.predict(x_te)
        y_prob = model.predict_proba(x_te)
        # keep probability for the positive class only
        y_prob = y_prob[:, 1]
        f1_s = f1_score(yy_test[:, i], y_pred)
        auroc_s = roc_auc_score(yy_test[:, i], y_prob)
        precision, recall, thresholds = precision_recall_curve(yy_test[:, i], y_prob)
        auprc_s = auc(recall, precision)
        ap50_s = ap50(y_prob, yy_test[:, i])

        f1_scores.append(f1_s)
        auroc_scores.append(auroc_s)
        auprc_scores.append(auprc_s)
        ap50_scores.append(ap50_s)
        frequency.append(yy_test[:, i].sum() / len(yy_test[:, i]))

    return f1_scores, auroc_scores, auprc_scores, frequency


def training_with_split(model, x, y):
    f1_scores = list()
    auroc_scores = list()
    auprc_scores = list()
    ap50_scores = list()
    frequency = list()

    for i in range(y.shape[1]):
        print(i)
        # split into train and test
        x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y[:, i])

        # here there should be a condition that would only keep one negative sample for
        # each triplet and not feed the model with all the unknown triplets as false
        pos_ind = y_tr[:, i].nonzero()
        pos_ind = pos_ind[0]
        rand_neg_ind = pos_ind + 1
        tr_index = np.concatenate((pos_ind, rand_neg_ind))
        tr_index = tr_index[:-1]

        pos_ind_t = y_te[:, i].nonzero()
        pos_ind_t = pos_ind_t[0]
        rand_neg_ind_t = pos_ind_t + 1
        te_index = np.concatenate((pos_ind_t, rand_neg_ind_t))
        te_index = te_index[:-1]

        model.fit(x_tr[tr_index, :], y_tr[tr_index, i])
        y_pred = model.predict(x_te[te_index, :])
        y_prob = model.predict_proba(x_te[te_index, :])
        # keep probability for the positive class only
        y_prob = y_prob[:, 1]
        f1_s = f1_score(y_te[te_index, i], y_pred)
        auroc_s = roc_auc_score(y_te[te_index, i], y_prob)
        precision, recall, thresholds = precision_recall_curve(y_te[te_index, i], y_prob)
        auprc_s = auc(recall, precision)
        ap50_s = ap50(y_prob, y_te[te_index, i])
        f1_scores.append(f1_s)
        auroc_scores.append(auroc_s)
        auprc_scores.append(auprc_s)
        ap50_scores.append(ap50_s)
        frequency.append(y_te[te_index, i].sum() / len(y_te[te_index, i]))

    return f1_scores, auroc_scores, auprc_scores, ap50_scores, frequency


def training_with_split_new(model, data):
    f1_scores = list()
    auroc_scores = list()
    auprc_scores = list()
    ap50_scores = list()
    frequency = list()

    relations = set(data['se'])

    for i, rel in enumerate(relations):
        print(i)
        full = data[data['se'] == rel]
        y = full['label']
        x = full[['node1', 'node2']].to_numpy().tolist()

        # one hot encode dataset and targets
        #y = MultiLabelBinarizer().fit_transform(labels)
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


def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : estimator instance
        An estimator instance implementing `fit` and `predict` methods which
        will be cloned for each validation.

    title : str
        Title for the chart.

    X : array-like of shape (n_samples, n_features)
        Training vector, where ``n_samples`` is the number of samples and
        ``n_features`` is the number of features.

    y : array-like of shape (n_samples) or (n_samples, n_features)
        Target relative to ``X`` for classification or regression;
        None for unsupervised learning.

    axes : array-like of shape (3,), default=None
        Axes to use for plotting the curves.

    ylim : tuple of shape (2,), default=None
        Defines minimum and maximum y-values plotted, e.g. (ymin, ymax).

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like of shape (n_ticks,)
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the ``dtype`` is float, it is regarded
        as a fraction of the maximum size of the training set (that is
        determined by the selected validation method), i.e. it has to be within
        (0, 1]. Otherwise it is interpreted as absolute sizes of the training
        sets. Note that for classification the number of samples usually have
        to be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True,
                       scoring='f1')
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt
