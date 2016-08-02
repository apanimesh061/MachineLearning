from __future__ import division
import numpy as np


def step_function(val):
    return -1 if val < 0 else 1


data = np.loadtxt('Data\\perceptronData.txt', delimiter='\t', dtype='float')

X = data[:, range(data.shape[1] - 1)]
y = data[:, [-1]]

mu_mat = X.mean(axis=0)
std_mat = X.std(axis=0)
X = (X - mu_mat) / std_mat  # normalize by zero mean and unit variance

# add bias
X_b = np.c_[X, np.ones(y.size)]
lr = 0.0001


def train_perceptron(matrix, target, learning_rate, max_iter=3000):
    no_of_instances, no_of_features = matrix.shape
    w = np.zeros(no_of_features)  # initialise weight to 0's
    no_of_iter = 0
    while True:
        mis_classifications = 0
        for instance, label in zip(matrix, target):
            predicted_label = step_function(np.vdot(instance, w))
            if predicted_label == label[0]:
                continue
            mis_classifications += 1
            w += (learning_rate * (label - predicted_label)) * instance
        print "Iteration {0}: Total mistakes {1}".format(no_of_iter + 1, mis_classifications)
        no_of_iter += 1
        if no_of_iter == max_iter or mis_classifications == 0:
            break
    return w


w = train_perceptron(X_b, y, lr)
print "Classifier weights: {0}".format(w)
normalizer = -w[-1]
norm_w = w[:-1] / normalizer
print "Normalized classifier weights: {0}".format(norm_w)

correct = 0
for (row, label) in zip(X, y):
    if step_function(np.vdot(row, norm_w)) == label[0]:
        correct += 1

print "Accuracy:", correct * 100 / y.size
