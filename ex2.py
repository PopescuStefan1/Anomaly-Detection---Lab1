import sklearn
import pyod
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
from pyod.models.knn import KNN

# Ex 2

X_train, X_test, y_train, y_test = pyod.utils.data.generate_data(400, 100, 2, 0.1)

knn = KNN(0.1)

knn.fit(X_train)

y_train_pred = knn.predict(X_train)
y_train_scores = knn.decision_function(X_train)
y_test_pred = knn.predict(X_test)
y_test_scores = knn.decision_function(X_test)

print("y_test_scores:", y_test_scores)

cm = sklearn.metrics.confusion_matrix(y_train, y_train_pred)
print(cm)

TP = cm[0][0]
TN = cm[1][1]
FN = cm[0][1]
FP = cm[1][0]

bal_acc = (TP / (TP + FN) + TN / (TN + FP)) / 2
print(bal_acc)

fpr, tpr, _ = sklearn.metrics.roc_curve(y_test, y_test_scores)
print("FPR: ", fpr, "TPR:", tpr)
plt.plot(fpr, tpr)
plt.show()
