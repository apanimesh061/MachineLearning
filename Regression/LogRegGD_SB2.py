from __future__ import division
from collections import defaultdict
import numpy as np
import math


data = np.loadtxt('Data\\spambase\\spambase.data', delimiter=',', dtype='float')
X = data[:, range(data.shape[1] - 1)]
y = data[:, [-1]]
mu_mat = X.mean(axis=0)
std_mat = X.std(axis=0)
X = (X - mu_mat) / std_mat
X = np.c_[X, np.ones(y.size)]

