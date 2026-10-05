import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('./datasets/ex1data1.txt', header = None, delimiter = ",") #read from dataset
X = data.iloc[:,0] # read first column, will be put in a 'series' variable
print('X.shape: ', X.shape)
y = data.iloc[:,1] # read second column, will be put in a 'series' variable
print('y.shape: ', y.shape)
m = len(y) # number of training example (97)
print('Number of samples:', m)
print(data.head()) # view first few rows of the data
 

plt.scatter(X, y)
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.show()

print(type(X)) # should be a pd.Series()
X = X.to_numpy()[:,np.newaxis] # convert pd.Series() to an np.ndarray
y = y.to_numpy()[:,np.newaxis] # convert pd.Series() to an np.ndarray

# Will generate a warning:
# Support for multi-dimensional indexing (e.g. `obj[:, None]`) is deprecated and will be removed in a future version.  Convert to a numpy array before indexing instead.

# # Better w/o warning
# X = X.to_numpy()[:,np.newaxis] # convert pd.Series() to an np.ndarray
# y = y.to_numpy()[:,np.newaxis] # convert pd.Series() to an np.ndarray

theta = np.zeros([2,1]) # start off with a (0, 0) array
iterations = 10000 # 
learning_rate = 0.01 # magic number, defines the increase in change in Theta in every iteration we take

# apply the trick we saw in the classroom of 'all ones' for x0, so that we can use matrix multiplication
# TAKE CARE WITH THE VARIABLE NAMES 
ones = np.ones([m,1]) # create a column vector of ones
X_ones = np.hstack((ones, X)) # add the column of ones to the original X data, so that we can use matrix multiplication


def computeCost(X_ones, y, theta):

    cost = np.dot(X_ones, theta) - y # calculate the difference between the predicted and actual values
    return np.sum(np.power(cost, 2)) / (2 * m) # calculate the cost function

def gradientDescent(X_ones, y, theta, learning_rate, iterations):

    # First work out what the partial derivatives are, then apply the gradientDescent 
    # Parameters:
    # -----------
    # X: matrix with the input variables
    # y: the correct results
    # iterations: The number of times the gradientDescent should run.
    #
    # FILL IN THE NECESSARY CODE.
    # After the iterations are performed, return the theta0 and theta1 values.
    #
    #
    #¨
    for i in range(iterations):

        cost = np.dot(X_ones, theta) - y
        gradient = np.dot(X_ones.T, cost) * (1/m)
        theta = theta - (learning_rate * gradient)

    return theta

cost = computeCost(X_ones, y, theta)
print(cost)
theta_refined = gradientDescent(X_ones, y, theta, learning_rate, iterations)
print(theta_refined)
cost_refined =  computeCost(X_ones, y, theta_refined)
print(cost_refined)
