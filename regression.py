# Aaron Barnett

import math
import numpy as np
import matplotlib.pyplot as plt


class Regressor:
    def __init__(self, degree):
        self.degree = 1
        if degree == 1:
            self.weights = np.array([0.1303863, 0.05355882])  # starting weights for polynomial degree one
            # self.weights = np.random.rand(2,)
        elif degree == 2:
            # self.weights = np.random.rand(3,)
            self.weights = np.array([0.22115549, 0.2138444, 0.26373176])  # starting weights for polynomial degree two
        elif degree == 4:
            # self.weights = np.random.rand(5,)
            self.weights = np.array([-0.54140065,  0.48168128, -0.27298078, -0.03292346, -0.09983796])  # starting weights for polynomial degree 4
        elif degree == 7:
            # self.weights = np.random.rand(8,)
            self.weights = np.array([-1.06850255,  1.2811347,  -0.18763978,  1.35401293 , 0.10015698 , 0.69093328 ,0.00479627, -0.39636938])
            # self.weights = np.array([-0.06337198, 0.07073019, -0.04089214, 0.06482378, -0.03962897, 0.05469127, -0.0191296, -0.05641042]) synth2
        self.alpha = 0.000001
        self.epochs = 20000

    def fit(self, x, y):
        for i in range(self.epochs):
            bias = self.weights[:1]
            weights = self.weights[1:]
            # append column of 1's to beginning of data to fold special bias case in
            b_error = np.sum(np.subtract(self.predict(x), y))
            bias = bias - self.alpha * (1/len(x)) * b_error
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
    file_index = 1
    data = np.genfromtxt(files[file_index], delimiter=',')
    x = data[:, 0]
    y = data[:, 1]
    for degree in [1,2,4,7]:
        x_poly = polynomial_features(x, degree)  # calculate basis expansion
        poly_regressor = Regressor(degree)  # initialize regressor
        poly_regressor.fit(x_poly, y)  # train regressor
        poly_regressor.plot(x, y, files[file_index])  # plot training data and curve
        print(poly_regressor.MSE(x_poly, y))  # calculate mean squared error
        print(poly_regressor.weights)


test_poly_regressor()
