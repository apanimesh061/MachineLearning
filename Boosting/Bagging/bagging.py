# ------------------------------------------------------------------------------
# Name:        bagging
# Purpose:
#
# Author:      Animesh Pandey
#
# Created:     01/11/2015
# Copyright:   (c) Animesh Pandey 2015
# ------------------------------------------------------------------------------

import numpy as np
from tabulate import tabulate
from DecisionTreeClassifier import DecisionTree


class Data(object):
    data = None
    X = None
    y = None
    no_of_instances = None
    vocab_size = None

    def __init__(self, path):
        self.data = np.loadtxt(path, delimiter=',', dtype="float")
        self.X = self.data[:, range(self.data.shape[1] - 1)]
        self.y = self.data[:, [-1]]
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

    def generate_randon_samples(self, max_iter):
        training_len = round(0.75 * self.no_of_instances)
        testing_len = self.no_of_instances - training_len
        for k in xrange(max_iter):
            yield np.random.choice(range(self.no_of_instances), training_len, replace=True), \
                  np.random.choice(range(self.no_of_instances), testing_len, replace=True)


if __name__ == "__main__":
    data = Data(path="../Data/spambase.data")
    avg_test_accuracy = 0.0
    avg_train_accuracy = 0.0
    no_of_episodes = 50
    table = []

    for current_fold, (training_indices, testing_indices) in enumerate(
            data.generate_randon_samples(max_iter=no_of_episodes)):
        training_set = data.X[training_indices]
        testing_set = data.X[testing_indices]

        training_target = data.y[training_indices]
        testing_target = data.y[testing_indices]

        dec_tree = DecisionTree(training_set, training_target)
        dec_tree.train()
        current_accuracy, current_accuracy_training = \
            dec_tree.get_accuracy(testing_set, testing_target)

        avg_test_accuracy += current_accuracy
        avg_train_accuracy += current_accuracy_training

        table.append([current_fold + 1, current_accuracy, current_accuracy_training])

    table.append(["Average", avg_test_accuracy / no_of_episodes, avg_train_accuracy / no_of_episodes])

    print tabulate(table, headers=["Episode", "Test Accuracy", "Train Accuracy"], tablefmt="psql")
