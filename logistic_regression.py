import numpy as np
import pdb

def hypothesis(x, theta):
  return sigmoid(np.dot(x, theta))

def sigmoid(z):
  return 1 / (1 + np.exp(-z))

def compute_cost(prediction, y, m):
  loss = np.subtract(prediction, y)
  cost = np.sum(loss ** 2) / (2 * m)
  return loss, cost

def gradientDescent(x, y, theta, alpha, m, numIterations):
    xTrans = x.transpose()
    for i in range(0, numIterations):
        prediction = hypothesis(x, theta)
        loss, cost = compute_cost(prediction, y , m)
        # print("Iteration %d | Cost: %f" % (i, cost))
        # avg gradient per example
        gradient = np.dot(xTrans, loss) / m
        # update
        theta = theta - alpha * gradient
    return theta

def train(x, y):
  alpha = 0.00001
  numIterations = 100000
  np_x = np.array(x)
  np_y = np.array(y)
  r,c = np_x.shape
  theta = np.zeros(c)
  return gradientDescent(np_x, np_y, theta, alpha, r, numIterations)
