from __future__ import division
from collections import defaultdict
from numpy.linalg import pinv
from ReadData import KFoldData
import numpy as np
from itertools import izip
import gc
from ROC import ROC

fold_size = 10

training_data = "Data\spambase\spambase.data"
feature_list = ['word_freq_make', 'word_freq_address', 'word_freq_all',
                'word_freq_3d', 'word_freq_our', 'word_freq_over', 'word_freq_remove',
                'word_freq_internet', 'word_freq_order', 'word_freq_mail', 'word_freq_receive',
                'word_freq_will', 'word_freq_people', 'word_freq_report',
                'word_freq_addresses', 'word_freq_free', 'word_freq_business',
                'word_freq_email', 'word_freq_you', 'word_freq_credit', 'word_freq_your',
                'word_freq_font', 'word_freq_000', 'word_freq_money', 'word_freq_hp',
                'word_freq_hpl', 'word_freq_george', 'word_freq_650', 'word_freq_lab',
                'word_freq_labs', 'word_freq_telnet', 'word_freq_857', 'word_freq_data',
                'word_freq_415', 'word_freq_85', 'word_freq_technology', 'word_freq_1999',
                'word_freq_parts', 'word_freq_pm', 'word_freq_direct', 'word_freq_cs',
                'word_freq_meeting', 'word_freq_original', 'word_freq_project', 'word_freq_re',
                'word_freq_edu', 'word_freq_table', 'word_freq_conference', 'char_freq_;',
                'char_freq_(', 'char_freq_[', 'char_freq_!', 'char_freq_$', 'char_freq_#',
                'capital_run_length_average', 'capital_run_length_longest',
                'capital_run_length_total']


def mat_stat(arr):
    mu_mat = arr.mean(axis=0)
    std_mat = arr.std(axis=0)
    return mu_mat, std_mat

def mean_shift_normalisation(matrix, mean, std):
    matrix = (matrix - mean) / std
    return matrix

dataset = KFoldData(training_data, {index: feature_name for (index, feature_name) in enumerate(feature_list)})
dataset.get_data_set()

def stream_datsets():
    for a, b in dataset.k_fold_cross_validation(K=fold_size, randomise=True):
        mean, var =  mat_stat(dataset.full_data[a])
        train_data = mean_shift_normalisation(dataset.full_data[a], mean, var)
        test_data = mean_shift_normalisation(dataset.full_data[b], mean, var)
##        train_data = dataset.full_data[a]
##        test_data = dataset.full_data[b]
        yield train_data.tolist(), dataset.target[a].tolist(), test_data.tolist(), dataset.target[b].tolist()

def stream_datasets():
    current_train_matrix = []
    current_train_target = []
    current_test_matrix = []
    current_test_target = []
    for a, b in dataset.k_fold_cross_validation(K=fold_size, randomise=True):
        for index in a:
            current_train_matrix.append(dataset.full_data[index])
            current_train_target.append(dataset.target[index])
        for index in b:
            current_test_matrix.append(dataset.full_data[index])
            current_test_target.append(dataset.target[index])
        yield current_train_matrix, current_train_target, current_test_matrix, current_test_target
        current_train_matrix = []
        current_train_target = []
        current_test_matrix = []
        current_test_target = []
##exit()

def get_accuracy():
    for trainer, trainer_target, tester, tester_target in stream_datsets():
        X, y, test_X, test_y = trainer, trainer_target, tester, tester_target
        X = np.matrix(X)
        y = np.matrix(y)
        X = np.c_[X, np.ones(y.size)]  # added bias to training set
        exp1 = X.transpose() * X
        idmat = np.eye(X.shape[1])
        possible_lambdas = np.linspace(-30, 30, 100)  # these are the possible lambdas
        best_acc = 0.0
        best_rc = None
        for ridge_constant in possible_lambdas:
            A = pinv(exp1 + (ridge_constant * idmat))
            B = X.transpose() * y.transpose()
            w = A * B
            solution = dict()
            sol = []
            for name, val in izip(feature_list, w):
                sol.append(val[0, 0])
                solution[name] = val[0, 0]

            test_X = np.matrix(test_X)
            test_y = np.array(test_y)
            test_X = np.c_[test_X, np.ones(test_y.size)]  # added bias to testing set
            no_of_matches = 0
            test_size = test_y.size
            for (vec, tar) in izip(np.array(test_X), test_y):
                prediction = sum(i * j for (i, j) in zip(vec, sol))
                pred_class = 0 if prediction <= 0 else 1
##                print prediction, tar, pred_class
                if tar == pred_class:
                    no_of_matches += 1
            curr_acc = no_of_matches / test_size

            no_of_matches = 0
            for (vec, tar) in izip(np.array(X), y.transpose()):
                prediction = sum(i * j for (i, j) in zip(vec, sol))
                pred_class = 0 if prediction <= 0 else 1
                if tar == pred_class:
                    no_of_matches += 1
            curr_acc_train = no_of_matches / y.size

            if curr_acc > best_acc:
                best_acc = curr_acc
                best_rc = ridge_constant
                best_train_acc = curr_acc_train

        yield best_acc, best_rc, best_train_acc


if __name__ == "__main__":
    total = 0.0
    for accuracy, rc, train_acc in get_accuracy():
        print accuracy, train_acc, rc
        total += accuracy
        exit()
    print "Average accuracy is {0}%".format(total * 100 / fold_size)
