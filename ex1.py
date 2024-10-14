import pyod
import matplotlib.pyplot as plt

# Ex 1

X_train, X_test, y_train, y_test = pyod.utils.data.generate_data(400, 100, 2, 0.1)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

plt.scatter(X_train[:-40, 0], X_train[:-40, 1], c='b')
plt.scatter(X_train[-40:, 0], X_train[-40:, 1], c='r')

plt.show()
