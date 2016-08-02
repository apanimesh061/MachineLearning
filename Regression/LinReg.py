from numpy.linalg import pinv
from scipy import sparse
from scipy.linalg import pinv as spinv
from ReadData import Data
import numpy as np
from itertools import izip
import matplotlib.pylab as plt

training_data = "Data\housing_train.txt"
test_data = "Data\housing_test.txt"

feature_list = ['CRIM',
                'ZN',
                'INDUS',
                'CHAS',
                'NOX',
                'RM',
                'AGE',
                'DIS',
                'RAD',
                'TAX',
                'PTRATIO',
                'B',
                'LSTAT',
                'bias']

dataset = Data(
        training_data,
        test_data,
        {index: feature_name for (index, feature_name) in enumerate(feature_list)}
)


# X, y = dataset.get_data_set(normalize=False)
# X = np.matrix(X)
# y = np.matrix(y)
# X = np.c_[X, np.ones(y.size)]  # bias added
# A = pinv((X.transpose() * X) + (ridge_constant * np.eye(X.shape[1])))

def get_residues():
    X, y = dataset.get_data_set(normalize=False)
    X = np.matrix(X)
    y = np.matrix(y)
    X = np.c_[X, np.ones(y.size)]  # bias added
    exp1 = X.transpose() * X
    idmat = np.eye(X.shape[1])
    possible_lambdas = np.linspace(-30, 30, 100)  # these are the possible lambdas
    best_mse = +1e039
    best_rc = None
    for ridge_constant in possible_lambdas:
        A = pinv(exp1 + (ridge_constant * idmat))
        B = X.transpose() * y.transpose()
        w = A * B
        solution = dict()
        sol = []
        for name, val in zip(feature_list, w):
            sol.append(val[0, 0])
            solution[name] = val[0, 0]
        housing_test_dat, actual_target = dataset.get_test_set(normalize=False)
        housing_test_dat = np.c_[np.matrix(housing_test_dat), np.ones(actual_target.size)]  # bias added
        test_sse = 0.0
        train_sse = 0.0
        for (vec, tar) in izip(np.array(housing_test_dat), actual_target):
            prediction = sum(i * j for (i, j) in zip(vec, sol))
            test_sse += (tar - prediction) ** 2

        for (vec, tar) in izip(np.array(X), y.transpose()):
            prediction = sum(i * j for (i, j) in zip(vec, sol))
            train_sse += (tar[0, 0] - prediction) ** 2

        current_mse = test_sse / actual_target.size
        if current_mse < best_mse:
            best_mse = current_mse
            best_rc = ridge_constant

    return best_mse, best_rc


if __name__ == "__main__":
    train_mse_list = []
    test_mse_list = []
    lam_list = []
    m = -10
    f = tuple()
    best_mse, best_rc = get_residues()
    print "Train MSE: {0} with lambda as {1}".format(best_mse, best_rc)

    # for train_mse, test_mse, lam in get_residues():
    #     if test_mse > m:
    #         f = (test_mse, lam)
    #     train_mse_list.append(train_mse)
    #     test_mse_list.append(test_mse)
    #     lam_list.append(lam)
    #     print "Train MSE: {0} with lambda as {1}".format(train_mse, lam)
    #
    # plt.plot(lam_list, train_mse_list, 'r-', lam_list, test_mse_list, 'b-')
    # plt.plot(lam_list, test_mse_list, 'b-')
    # plt.axis([-30, 30, 0, 50])
    # plt.grid()
    # plt.show()
