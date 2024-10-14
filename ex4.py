import sklearn
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import sklearn.metrics

from sklearn.metrics import balanced_accuracy_score

mean = [0, 0, 0]
cov = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

n_samples = 900
X_normal = np.random.multivariate_normal(mean, cov, n_samples)

# print(X_normal)

fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")
ax.scatter3D(X_normal[:, 0], X_normal[:, 1], X_normal[:, 2], c = "b")
 
n_outliers = 100
mean_anomalies = [2, 2, 2] 
cov_anomalies = [[10, 0, 0], [0, 10, 0], [0, 0, 10]]  
X_outliers = np.random.multivariate_normal(mean_anomalies, cov_anomalies, n_outliers)

# print(X_outliers)

ax.scatter3D(X_outliers[:, 0], X_outliers[:, 1], X_outliers[:, 2], c = "r")
# plt.show()

X = np.vstack((X_normal, X_outliers))

# print(X)

y_true = np.hstack((np.zeros(n_samples), np.ones(n_outliers)))

# print(y_true)

z_scores = stats.zscore(X, axis=0)

# print("Z-scores: ", z_scores)

threshold = np.quantile(np.abs(z_scores), 1 - 0.1)

# print('Threshold:', threshold)

y_pred = (np.abs(z_scores).max(axis=1) > threshold).astype(int)

bal_acc = balanced_accuracy_score(y_true, y_pred)

print(bal_acc)
