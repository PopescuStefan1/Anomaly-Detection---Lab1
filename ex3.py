import sklearn
import pyod
import scipy.stats as stats
import numpy as np

# Ex 3

X_train, X_test, y_train, y_test = pyod.utils.data.generate_data(1000, 0, 1, 0.1)

z_scores = stats.zscore(X_train)

threshold = np.quantile(np.abs(z_scores), 1 - 0.1)

print('Threshold:', threshold)

y_pred = (np.abs(z_scores) > threshold).astype(int)

bal_acc = sklearn.metrics.balanced_accuracy_score(y_train, y_pred)

print(bal_acc)
