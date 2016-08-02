# ------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      Animesh Pandey
#
# Created:     01/11/2015
# Copyright:   (c) Animesh Pandey 2015
# ------------------------------------------------------------------------------

from sklearn.datasets import load_boston
import numpy as np
from tabulate import tabulate
from DecisionTreeRegressor import RegressionTree


class Data(object):
    data = None
    X = None
    y = None
    no_of_instances = None
    vocab_size = None

    def __init__(self):
        self.data = load_boston()
        self.X = self.data["data"]
        self.y = self.data["target"]
        self.no_of_instances, self.vocab_size = self.X.shape

    def generate_k_folds(self, K=10, randomise=True):
        matrix = [i for i in xrange(len(self.y))]
        if randomise:
            from random import shuffle
            matrix = list(matrix)
            shuffle(matrix)
        for k in xrange(K):
            training = [x for i, x in enumerate(matrix) if i % K != k]
            validation = [x for i, x in enumerate(matrix) if i % K == k]
            yield np.array(training), np.array(validation)


def rse(x, y):
##    mean = np.mean(y, axis=0)
    val = 0.0
    for i, j in zip(x, y):
        val += (i - j)**2
    return val / y.size


if __name__ == "__main__":
    data = Data()
    avg_test_accuracy = 0.0
    avg_train_accuracy = 0.0
    table = []
    total_test_mse = total_train_mse = 0.0
    for k, (training_indices, testing_indices) in enumerate(data.generate_k_folds()):
        training_set = data.X[training_indices]
        testing_set = data.X[testing_indices]
        training_target = data.y[training_indices]
        testing_target = data.y[testing_indices]
        current_target = training_target
        models = []
        for _ in range(50):
            reg_tree = RegressionTree(training_set, current_target)
            reg_tree.train(1)
            models.append(reg_tree)
            labels = reg_tree.test(training_set, current_target)
            current_target = current_target - labels

        final_test = np.zeros(testing_target.size)
        final_train = np.zeros(training_target.size)
        for model in models:
            final_test += model.test(testing_set, testing_target)
            final_train += model.test(training_set, training_target)

        current_mse_test = rse(final_test, testing_target)
        current_mse_train = rse(final_train, training_target)
        total_test_mse += current_mse_test
        total_train_mse += current_mse_train

        table.append([k + 1, current_mse_test, current_mse_train])

    table.append(["Average", total_test_mse / 10.0, total_train_mse / 10.0])

    print tabulate(table, headers=["Current Fold", "Test Error", "Train Error"], tablefmt="psql")
