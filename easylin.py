import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('./datasets/ex1data2.txt', header=None, delimiter=",")
print(dataset.head(5))

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
m = len(y)
print(X.shape, y.shape, m) 


mu = np.mean(X, axis=0)
sigma = np.std(X, axis=0)
X_norm = (X - mu) / sigma

print("Mean of X:", mu)
print("Normalized X (first 5 rows):\n", X_norm[:5])


X_aug = np.hstack((np.ones((m, 1)), X_norm))   # shape (m, 3)

learning_rate = 0.01
iterations = 400
theta = np.zeros(X_aug.shape[1])              

def computeCostMulti(X, y, theta):
    m = len(y)
    errors = X @ theta - y
    return (1 / (2 * m)) * np.sum(errors ** 2)

cost = computeCostMulti(X_aug, y, theta)
print("Initial cost:", cost)   # about 6.559e10 for theta = 0

def gradientDescentMulti(X, y, theta, learning_rate, iterations):
    m = len(y)
    theta = theta.copy()
    J_history = []

    for _ in range(iterations):
        errors = X @ theta - y
        gradient = (1 / m) * (X.T @ errors)
        theta = theta - learning_rate * gradient
        J_history.append(computeCostMulti(X, y, theta))

    return theta, J_history


theta, J_history = gradientDescentMulti(X_aug, y, theta, learning_rate, iterations)
print("Theta found by gradient descent:", theta)
print("Final cost:", J_history[-1])

plt.plot(range(len(J_history)), J_history)
plt.xlabel('Iteration')
plt.ylabel('Cost J')
plt.title('Convergence of gradient descent')
plt.show()

x_new = (np.array([1650, 3]) - mu) / sigma
x_new = np.hstack(([1], x_new))
price = x_new @ theta
print("Predicted price for a 1650 sq ft, 3-bedroom house:", price)


y_pred = X_aug @ theta
plt.scatter(y, y_pred, color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel('Actual price')
plt.ylabel('Predicted price')
plt.show()