import math
import numpy as np
import matplotlib.pyplot as plt


class Regressor:
    def __init__(self, degree):
        self.degree = 1
        if degree == 1:
            self.weights = np.array([0.05, 0.05])  # starting weights for polynomial degree one
        elif degree == 2:
            self.weights = np.array([1, 0, 1])  # starting weights for polynomial degree two
        elif degree == 4:
            self.weights = np.array([1, 0, 1, 0, 1])  # starting weights for polynomial degree 4
        elif degree == 7:
            self.weights = np.array([0.97317577, 0.15005726, 0.59010096, 0.22459232, 0.57386271, -0.02791413, -0.05072404, -0.00916566])
            # starting weights for polynomial degree 7 (MSE 4.6542 on dataset1), fit is WEIRD tho
        self.alpha = 0.000001
        self.epochs = 20000

    def fit(self, x, y):
        for i in range(self.epochs):
            bias = self.weights[:1]
            weights = self.weights[1:]
            # append column of 1's to beginning of data to fold special bias case in
            b_error = np.sum(np.subtract(self.predict(x), y))
            bias = bias - self.alpha * (1/len(x)) * b_error
            # w_error = np.sum(np.matmul(np.subtract(self.predict(x), y), x))
            for i in range(len(weights)):
                weights[i] = weights[i] - self.alpha * (1 / len(x)) * np.sum(np.matmul(np.subtract(self.predict(x), y), x[:, i]))
            self.weights = np.concatenate((bias, weights))

    def predict(self, x):
        bias = self.weights[:1]
        weights = self.weights[1:]
        if np.isscalar(x):
            y_pred = bias + np.dot(weights, x)
            return y_pred
        else:
            y_pred = np.zeros(len(x))
            for i in range(len(x)):
                y_pred[i] = bias + np.dot(weights, x[i])
            return y_pred

    def plot(self, x, y, title):
        plt.scatter(x, y, None, "b")
        curve = self.weights
        y_orig = y
        x_orig = x
        x = np.linspace(min(x) - 1, max(x) + 1, 100)
        y = [np.polyval(curve, i) for i in x]
        plt.plot(x, y, 'r')
        # change max and min to readjust graph
        plt.xlim(min(x_orig) - 1, max(x_orig) + 1)
        plt.ylim(min(y_orig) - 1, max(y_orig) + 1)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(title)
        plt.grid()
        plt.show()

    def MSE(self, x, y):
        bias = self.weights[:1]
        weights = self.weights[1:]
        y_pred = np.zeros(len(x))
        for i in range(len(x)):
            y_pred[i] = bias + np.dot(weights, x[i])
        error = np.sum(np.power(np.subtract(y, y_pred), 2))
        return error/len(x)


def polynomial_features(x, degree):
    expansion = np.zeros((len(x), degree - 1))
    for i in range(len(x)):
        for j in range(0, degree - 1):
            expansion[i, j] = x[i] ** (j + 2)
    return np.c_[x.reshape(expansion.shape[0]), expansion]


def test_poly_regressor():
    files = ['data/synthetic-1.csv', 'data/synthetic-2.csv', 'data/synthetic-3.csv']
    data = np.genfromtxt(files[0], delimiter=',')
    x = data[:, 0]
    y = data[:, 1]
    degree = 7
    x_poly = polynomial_features(x, degree)  # calculate basis expansion
    poly_regressor = Regressor(degree)  # initialize regressor
    poly_regressor.fit(x_poly, y)  # train regressor
    poly_regressor.plot(x, y, files[0])  # plot training data and fit curve
    print(poly_regressor.MSE(x_poly, y))  # calculate mean squared error
    print(poly_regressor.weights)


test_poly_regressor()