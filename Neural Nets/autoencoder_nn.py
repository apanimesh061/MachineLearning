from __future__ import division
import numpy as np
import math


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_prime(x):
    """
    First order derivative of sigmoid function
    """
    return sigmoid(x) * (1.0 - sigmoid(x))


class NeuralNetwork(object):
    def __init__(self, layers):
        self.activation = sigmoid
        self.activation_prime = sigmoid_prime
        self.weights = []
        for i in range(1, len(layers) - 1):
            r = 2 * np.random.random((layers[i - 1] + 1, layers[i] + 1)) - 1
            self.weights.append(r)
        r = 2 * np.random.random((layers[i] + 1, layers[i + 1])) - 1
        self.weights.append(r)

    def fit(self, X, y, learning_rate=0.3, epochs=400000):
        ones = np.atleast_2d(np.ones(X.shape[0]))  # adding bias to the data
        X = np.concatenate((ones.T, X), axis=1)

        for k in range(epochs):  # run till the end
            i = np.random.randint(X.shape[0])
            a = [X[i]]

            for l in range(len(self.weights)):
                dot_value = np.dot(a[l], self.weights[l])
                activation = self.activation(dot_value)
                a.append(activation)
            # output layer
            error = y[i] - a[-1]
            deltas = [error * self.activation_prime(a[-1])]

            # we need to begin at the second to last layer
            # (a layer before the output layer)
            for l in range(len(a) - 2, 0, -1):
                deltas.append(deltas[-1].dot(self.weights[l].T) * self.activation_prime(a[l]))

            # reverse
            # [level3(output)->level2(hidden)]  => [level2(hidden)->level3(output)]
            deltas.reverse()

            # backpropagation
            # 1. Multiply its output delta and input activation
            #    to get the gradient of the weight.
            # 2. Subtract a ratio (percentage) of the gradient from the weight.
            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * layer.transpose().dot(delta)

            if k % 10000 == 0:
                print 'epochs:', k

    def predict(self, x):
        a = np.concatenate((np.ones(1).T, np.array(x)), axis=1)
        for l in range(0, len(self.weights)):
            a = self.activation(np.dot(a, self.weights[l]))
        return a


if __name__ == '__main__':

    nn = NeuralNetwork([8, 3, 8])
    print nn.weights
    X = np.eye(8)
    # X = np.array([[0, 0, 1],
    #               [0, 1, 0],
    #               [1, 0, 1],
    #               [1, 1, 0]])
    # y = np.array([0, 1, 1, 0])
    y = np.eye(8)
    nn.fit(X, y)
    for e in X:
        print(e, nn.predict(e))
    print nn.weights
