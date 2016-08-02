from __future__ import division
import numpy as np

data = np.loadtxt('Data\\housing_train.txt', delimiter='\\s+', dtype='str')
test_data = np.loadtxt('Data\\housing_test.txt', delimiter='\\s+', dtype='str')


def goodness(weights, matrix, target):
    sse = 0.0
    for row, val in zip(matrix, target):
        actual_value = val
        predicted_value = np.vdot(row, weights)
        sse += (predicted_value - actual_value)**2
    return sse / target.size


def create_set(full_data):
    matrix = []
    tar_col = []
    for line in full_data:
        new_line = [float(val) for val in line.split()]
        matrix.append(new_line[:-1])
        tar_col.append(new_line[-1])
    matrix = np.asarray(matrix)
    tar_col = np.asarray(tar_col)
    return matrix, tar_col


def stochiastic_gd(matrix, target, learning_rate, min_diff=0.00015, max_iter=1500):
    no_of_instances, no_of_features = matrix.shape
    w = np.zeros(no_of_features)
    grad_list = [(target[i] - np.dot(w, matrix[i])) * matrix[i] for i in xrange(no_of_instances)]
    w = w + learning_rate * np.sum(grad_list, axis=0)
    optimal_sse = np.sum([(target[i] - np.dot(w, matrix[i])) ** 2 for i in xrange(no_of_instances)], axis=0)
    no_of_iter = 0
    while True:
        grad_list = [(target[i] - np.dot(w, matrix[i])) * matrix[i] for i in xrange(no_of_instances)]
        w = w + learning_rate * np.sum(grad_list, axis=0)
        sub_optimal_sse = np.sum([(target[i] - np.dot(w, matrix[i])) ** 2 for i in xrange(no_of_instances)], axis=0)
        if abs(optimal_sse - sub_optimal_sse) <= min_diff:
            break
        # print optimal_sse, sub_optimal_sse, abs(optimal_sse - sub_optimal_sse) <= min_diff, abs(optimal_sse - sub_optimal_sse)
        optimal_sse = sub_optimal_sse
        no_of_iter += 1
        if no_of_iter == max_iter:
            break
    return w


# training and test sets
X, y = create_set(data)
X_test, y_test = create_set(test_data)

# normalize the dataset
mu_mat = np.vstack((X, X_test)).mean(axis=0)
std_mat = np.vstack((X, X_test)).std(axis=0)
X = (X - mu_mat) / std_mat
X_test = (X_test - mu_mat) / std_mat

# Adding bias to the sets
X = np.c_[X, np.ones(y.size)]
X_test = np.c_[X_test, np.ones(y_test.size)]

# run gradient descent
final_weights = stochiastic_gd(X, y, 0.0001)

print final_weights
print goodness(final_weights, X_test, y_test)
print goodness(final_weights, X, y)