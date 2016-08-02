from __future__ import division
import numpy as np
import time
from ROC import ROC


data = np.loadtxt('Data\\spambase\\spambase.data', delimiter=',', dtype='float')
X = data[:, range(data.shape[1] - 1)]
y = data[:, [-1]]


def mat_stat(arr):
    mu_mat = arr.mean(axis=0)
    std_mat = arr.std(axis=0)
    return mu_mat, std_mat


def mean_shift_normalisation(matrix, mean, std):
    matrix = (matrix - mean) / std
    return matrix


def goodness(weights, matrix, target):
    correct = 0
    for row, val in zip(matrix, target):
        predicted_value = np.vdot(row, weights)
        label = 1 if predicted_value > 0.4 else 0
        if val == label:
            correct += 1
    return correct * 100 / target.size


def performance(weights, matrix, target, roc_type):
    scores = []
    actual_label = []
    threshold = 0.4
    for row, val in zip(matrix, target):
        predicted_value = np.vdot(row, weights)
        scores.append(predicted_value)
        actual_label.append(val[0])

    roc = ROC(scores, actual_label, roc_type)
    roc.compute()
    roc.plot()


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


def stochiastic_gd(matrix, target, learning_rate, reg_factor, min_diff=0.0001, max_iter=5000):
    no_of_instances, no_of_features = matrix.shape
    # w = np.zeros(no_of_features)
    w = np.random.random(no_of_features)
    grad_list = [(target[i] - np.vdot(matrix[i], w)) * matrix[i] for i in xrange(no_of_instances)]
    w = w * (1 - (learning_rate * reg_factor)) + learning_rate * np.sum(grad_list, axis=0)
    optimal_sse = np.sum([(target[i] - np.vdot(matrix[i], w)) ** 2 for i in xrange(no_of_instances)], axis=0)
    no_of_iter = 0
    while True:
        grad_list = [(target[i] - np.vdot(matrix[i], w)) * matrix[i] for i in xrange(no_of_instances)]
        w = w * (1 - (learning_rate * reg_factor)) + learning_rate * np.sum(grad_list, axis=0)
        sub_optimal_sse = np.sum([(target[i] - np.vdot(matrix[i], w)) ** 2 for i in xrange(no_of_instances)], axis=0)
        # print optimal_sse, sub_optimal_sse, abs(optimal_sse - sub_optimal_sse) <= min_diff, abs(optimal_sse - sub_optimal_sse)
        if abs(optimal_sse - sub_optimal_sse) <= min_diff:
            break
##        print optimal_sse, sub_optimal_sse, optimal_sse - sub_optimal_sse
        optimal_sse = sub_optimal_sse
        no_of_iter += 1
        if no_of_iter == max_iter:
            break
    return w


for (a, b) in k_fold_cross_validation(X, y, randomise=True):
    print "Starting new..."
    current_train_indices, current_test_indices = a, b

    current_training_set = X[current_train_indices] # load the training data
    mean, var =  mat_stat(current_training_set) # get stats
    current_training_set = mean_shift_normalisation(current_training_set, mean, var)
    current_training_target = y[current_train_indices]
    current_training_set = np.c_[current_training_set, np.ones(current_training_target.size)]

    current_testing_set = X[current_test_indices]
    current_testing_set = mean_shift_normalisation(current_testing_set, mean, var)
    current_testing_target = y[current_test_indices]
    current_testing_set = np.c_[current_testing_set, np.ones(current_testing_target.size)]

    lr = 0.00007

    w = stochiastic_gd(current_training_set, current_training_target, lr, 0.7)
##    print goodness(w, current_training_set, current_training_target)
##    print goodness(w, current_testing_set, current_testing_target)
    performance(w, current_testing_set, current_testing_target, "Linear Regression (GD)")
