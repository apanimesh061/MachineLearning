from __future__ import division
import numpy as np
from tabulate import tabulate


# Gaussian Discriminant Analysis on spambase.data

class GDA(object):
    phi = None
    mu0 = None
    mu1 = None
    sigma = None

    def __init__(self):
        self.phi = None
        self.mu0 = None
        self.mu1 = None
        self.sigma = None

    def train(self, X, Y):
        no_of_instances, no_of_features = X.shape

        I0 = [index for index, y in enumerate(Y) if y == 0]
        I1 = [index for index, y in enumerate(Y) if y == 1]

        self.phi = len(I1) / no_of_instances
        self.mu0 = np.mean(X[I0], axis=0)
        self.mu1 = np.mean(X[I1], axis=0)

        r = np.matrix(np.zeros([no_of_instances, no_of_features]))
        r[I0] = X[I0] - self.mu0
        r[I1] = X[I1] - self.mu1

        self.sigma = (1.0 / no_of_instances) * (r.T * r)

    def test(self, X, Y):
        r0 = X - self.mu0
        r1 = X - self.mu1

        z0 = np.sum(np.multiply(r0 * self.sigma.I, r0), axis=1)
        z1 = np.sum(np.multiply(r1 * self.sigma.I, r1), axis=1)

        s = self.phi * np.exp(-0.5 * z1) - (1.0 - self.phi) * np.exp(-0.5 * z0)

        Y_pred = s.copy()
        Y_pred[Y_pred <= 0.0] = 0
        Y_pred[Y_pred > 0.0] = 1

        P = np.matrix(np.zeros(Y.shape))
        P[np.where(Y == Y_pred)] = 1

        return 1.0 * P.sum() / len(Y)


def k_fold_cross_validation(matrix, target, K=10, randomise=False):
    matrix = [i for i in xrange(len(target))]
    if randomise:
        from random import shuffle
        matrix = list(matrix)
        shuffle(matrix)
    for k in xrange(K):
        training = [x for i, x in enumerate(matrix) if i % K != k]
        validation = [x for i, x in enumerate(matrix) if i % K == k]
        yield training, validation


def mat_stat(arr):
    mu_mat = arr.mean(axis=0)
    std_mat = arr.std(axis=0)
    return mu_mat, std_mat


def mean_shift_normalisation(matrix, mean, std):
    matrix = (matrix - mean) / std
    return matrix


if __name__ == '__main__':

    data = np.loadtxt('../Data/spambase/spambase.data', delimiter=',', dtype='float')
    X = data[:, range(data.shape[1] - 1)]
    y = data[:, [-1]]

    table = []
    average_training_accuracy = 0.0
    average_testing_accuracy = 0.0
    for current_fold, (a, b) in enumerate(k_fold_cross_validation(X, y, randomise=True)):
        current_train_indices, current_test_indices = a, b

        current_training_set = X[current_train_indices]
        mean, var = mat_stat(current_training_set)
        # current_training_set = mean_shift_normalisation(current_training_set, mean, var)
        current_training_target = y[current_train_indices]

        current_testing_set = X[current_test_indices]
        # current_testing_set = mean_shift_normalisation(current_testing_set, mean, var)
        current_testing_target = y[current_test_indices]

        clf = GDA()
        clf.train(current_training_set, current_training_target)

        acc_train = clf.test(current_training_set, current_training_target)
        acc_test = clf.test(current_testing_set, current_testing_target)

        average_testing_accuracy += acc_test
        average_training_accuracy += acc_train

        table.append([current_fold + 1, str(100.0 * acc_train) + " %", str(100.0 * acc_test) + " %"])
    table.append(["Average", str(100.0 * average_training_accuracy / 10) + " %",
                  str(100.0 * average_testing_accuracy / 10) + " %"])

    print tabulate(table, headers=["Current Fold", "Training Accruacy", "Testing Accuracy"], tablefmt="psql")
