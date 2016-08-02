import numpy as np


def mean(fv):
    return float(sum(fv)) / len(fv)


def randomvariable(th):
    return lambda x: th < x


def distribution(fv, rv):
    no_of_successes = sum([rv(f) for f in fv])
    prob_success = float(no_of_successes + 1) / (len(fv) + 57)
    prob_failure = 1.0 - prob_success

    def distribution_helper(x):
        return prob_success if rv(x) else prob_failure
    return distribution_helper


def model(fv, th=None):
    fv = np.asarray(list(fv))
    fv = fv[~np.isnan(fv)]
    fv = fv.tolist()
    rv = randomvariable(mean(fv) if th is None else th)
    return distribution(fv, rv)
