import math
import numpy as np
import matplotlib.pyplot as plt


class Regressor:
    def __init__(self, degree):
        self.degree = 0
        if degree == 1:
            self.weights = np.array([0.05, 0.05])  # weights for polynomial degree one
        elif degree == 2:
            self.weights = np.array([0.05289785, 0.04852937, 0])  # weights for polynomial degree two
        elif degree == 4:
            self.weights = np.array([.001, .001, .001, .001, .001])  # weights for polynomial degree 4
        elif degree == 7:
            self.weights = np.array([.001, .001, .001, .001, .001, .001, .001, .001])  # weights for polynomial degree 7
        self.alpha = 0.000001
        self.epochs = 20000

    def fit(self, x, y):
        for i in range(self.epochs):
            bias = self.weights[:1]
            weights = self.weights[1:]
            # append column of 1's to beginning of data to fold special bias case in
            b_error = np.sum(np.subtract(self.predict(x), y))
            bias = bias - self.alpha * (1/len(x)) * b_error
            w_error = np.sum(np.matmul(np.subtract(self.predict(x), y), x))
            for i in range(len(weights)):
                weights[i] = weights[i] - self.alpha * (1 / len(x)) * w_error
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

    def plot(self, title):
        x = np.linspace(-2.5, 2.5, 100)
        y = self.predict(x)
        plt.plot(x, y, '-r')
        plt.xlim((-3, 3))
        plt.ylim((0, 10))
        plt.title(title)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend('upper left')
        plt.grid()
        plt.show()

    def MSE(self, x, y):
        error = 0
        bias = self.weights[:1]
        weights = self.weights[1:]
        y_pred = np.zeros(len(x))
        for i in range(len(x)):
            y_pred[i] = bias + np.dot(weights, x[i])
        error = np.sum(np.power(np.subtract(y, y_pred), 2))
        return error/len(x)


def polynomial_features(x, degree):
    added_x = np.zeros((len(x), degree - 1))
    for i in range(len(x)):
        for j in range(0, degree - 1):
            added_x[i, j] = x[i] ** (j + 2)
    return added_x


def test_poly_regressor():
    data = np.genfromtxt('data/synthetic-3.csv', delimiter=',')
    x = data[:, 0]
    y = data[:, 1]
    degree = 7
    expansion = polynomial_features(x, degree)  # calculate basis expansion
    x = np.c_[x.reshape(expansion.shape[0]), expansion]  # column stack x with expansions
    poly_regressor = Regressor(degree)  # initialize regressor
    poly_regressor.fit(x, y)  # train regressor
    # poly_regressor.plot("Synthetic-1")
    print(poly_regressor.MSE(x, y))  # calculate mean squared error
    print(poly_regressor.weights)


test_poly_regressor()