# import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.experimental import enable_hist_gradient_boosting
# from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.svm import SVC
# from sklearn.datasets import load_digits
# from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
# from sklearn.metrics import f1_score
from helpers import plot_learning_curve


fig, axes = plt.subplots(3, 2, figsize=(10, 15))

#X, y = load_digits(return_X_y=True)
x_curves = x
y_curves = y[:, 0]
title = "Learning Curves (Hist)"
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=1, test_size=0.2, random_state=0)

estimator = HistGradientBoostingClassifier()
plot_learning_curve(estimator, title, x_curves, y_curves, axes=axes[:, 0], ylim=(0.0, 1.01),
                    cv=cv)

title = r"Learning Curves (Bayes)"
# SVC is more expensive so we do a lower number of CV iterations:
cv = ShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
estimator = GaussianNB()
plot_learning_curve(estimator, title, x_curves, y_curves, axes=axes[:, 1], ylim=(0.0, 1.01),
                    cv=cv)

plt.show()
