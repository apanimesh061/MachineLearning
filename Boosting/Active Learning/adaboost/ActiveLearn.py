from __future__ import division
from decisionstumps import StumpLibrary, Stump
from boosting import Boosting
from resultset import *
import numpy as np


class DataPoint(object):
    features = None
    label = None

    def __init__(self, features, label):
        self.features = features
        if label == 0.0:
            self.label = -1.0
        else:
            self.label = label

    def __str__(self):
        return "{0} {1} labelled {2}".format(self.__class__.__name__, self.features, self.label)

    def __repr__(self):
        return "{0}({1}, {2})".format(self.__class__.__name__, self.features, self.label)


class Data(object):
    data = None
    X = None
    y = None
    no_of_instances = None
    vocab_size = None

    def __init__(self, path):
        self.data = np.loadtxt(path, delimiter=',', dtype="float")
        self.X = self.data[:, range(self.data.shape[1]-1)]
        self.y = self.data[:, [-1]]
        self.no_of_instances, self.vocab_size = self.X.shape

    def generate_k_folds(self, K=10, randomise=True, fold=0):
        matrix = self.X
        target = self.y
        matrix = [i for i in xrange(len(target))]
        if randomise:
            from random import shuffle
            matrix = list(matrix)
            shuffle(matrix)
        for k in xrange(K):
            training = [x for i, x in enumerate(matrix) if i % K != k]
            validation = [x for i, x in enumerate(matrix) if i % K == k]
            if k == fold:
                yield np.array(training), np.array(validation)
            else:
                continue


def to_data_point(matrix):
    new_data = []
    for row in matrix:
        feature = row[:-1]
        label = row[-1]
        new_data.append(DataPoint(feature, label))
    return new_data


class AdaBoost(object):
    # sets comprising of X and y
    training = None
    testing = None
    hypothesis = None
    booster = None
    weight_vector = None
    addition = None
    new_testing = None

    plotter = None

    def __init__(self, training, testing):
        self.training = training
        self.testing = testing

    def train(self):
        self.booster = Boosting(self.training)
        self.weight_vector = self.booster.init()
        self.hypothesis = self.booster.hypothesis
        print "Optimal hypothesis created"

    def classify(self):
        sv = StumpLibrary(self.training)
        svpick = sv.pick_best
        roundct = 0
        testerr = 1.0
        prev_err = testerr
        TABLE_ = \
            '+------------+---------------+----------------+---------------+'
        TABLE_HEADER = \
            '|   Round    | Local Error   | Training Error | Testing Error |'
        TABLE_ROW_FORMAT = \
            '| {: ^10d} |  {: <10.8f}   | {: <10.10f}   |  {: <10.8f}   |'

        print TABLE_
        print TABLE_HEADER
        print TABLE_
        while True:
            roundct += 1
            stump = svpick(self.weight_vector)

            current_local_error = sv.current_local_error
            curried = (lambda s: lambda dp: Stump.query(s, dp))(stump)
            self.weight_vector = self.booster.round(curried)

            roc_dp_pairs = [(dp, DataResult(int(dp.label > 0), self.booster.model(dp))) for dp in self.testing]

            current_round = roundct

            current_training_error = self.misclassification_count(self.training) / float(len(self.training))
            current_testing_error = self.misclassification_count(self.testing) / float(len(self.testing))

            if current_round == 40:
                close_points = map(lambda x: x[0], sorted(roc_dp_pairs, key=lambda x: abs(x[1].score)))
                to_be_added = close_points[: int(round(0.02 * len(close_points)))]
                to_be_tested = close_points[int(round(0.02 * len(close_points))):]
                self.addition = to_be_added
                self.new_testing = to_be_tested
                break
            else:
                prev_err = current_testing_error

            print TABLE_ROW_FORMAT.format(current_round, current_local_error,
                                          current_training_error, current_testing_error)
        print TABLE_

    def misclassification_count(self, dataset):
        count = 0
        for dp in dataset:
            if self.hypothesis(dp) != dp.label:
                count += 1
        return count
