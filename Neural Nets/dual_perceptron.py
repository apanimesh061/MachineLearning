from __future__ import division
import numpy as np
from sklearn import preprocessing
np.random.seed(123)


class Kernels:
    def __init__(self):
        raise NotImplementedError

    @staticmethod
    def polynomial(x, y, p=2):
        return (1 + np.dot(x, y)) ** p

    @staticmethod
    def rbf(x, y, sigma=2.0):
        gamma = 1 / (2 * (sigma ** 2))
        squared_euclidian = np.sum(np.square(np.subtract(y, x)), axis=1)
        return np.exp(-gamma*squared_euclidian)

    @staticmethod
    def dot_product(x, y):
        return np.dot(x, y)


class KernelPerceptron:
    def __init__(self, data_type, normalize=False):
        data = np.loadtxt(data_type, delimiter='\t', dtype=np.float32)
        np.random.shuffle(data)
        X = data[:, range(data.shape[1] - 1)]
        self.y = data[:, [-1]].transpose()[0]
        self.X = X
        self.normalize = normalize

    def train(self, kernel_type, max_iter=100):
        if self.normalize:
            self.X = preprocessing.scale(self.X)
        gram_matrix = np.empty((self.y.size, self.y.size), dtype=np.float32)
        for i in xrange(self.X.shape[0]):
            gram_matrix[i, :] = kernel_type(self.X, self.X[i])

        alpha = np.zeros(self.X.shape[0])
        current_iter = 0
        while True:
            for row in range(self.X.shape[0]):
                prediction_y = np.sign(np.dot(alpha, gram_matrix[:, row]))
                if prediction_y != self.y[row]:
                    alpha[row] = np.add(alpha[row], self.y[row])

            y_train = np.sign(np.dot(alpha, gram_matrix))

            if 0 in y_train:
                y_train[np.where(y_train == 0)] = 1

            num_mis_class = sum(y_train != self.y)
            print num_mis_class, "misses with", (1 - (num_mis_class / self.y.size)) * 100.0, "% accuracy"

            current_iter += 1
            if num_mis_class == 0 or current_iter == max_iter:
                break

if __name__ == '__main__':
    linearly_separable = "../Data/perceptronData.txt"
    linearly_non_separable = "../Data/twoSpirals.txt"

    # print "Testing linearly separable data with Dot Product kernel"
    # dual_percepton = KernelPerceptron(data_type=linearly_separable, normalize=True)
    # dual_percepton.train(kernel_type=Kernels.dot_product)

    # print
    # print "Testing linearly un-separable data with Dot Product kernel"
    # dual_percepton = KernelPerceptron(data_type=linearly_non_separable)
    # dual_percepton.train(kernel_type=Kernels.dot_product)
    #
    print
    print "Testing linearly un-separable data with RBF kernel"
    dual_percepton = KernelPerceptron(data_type=linearly_non_separable)
    dual_percepton.train(kernel_type=Kernels.rbf)
