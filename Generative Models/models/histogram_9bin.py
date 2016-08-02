import math
import collections
import numpy as np


def mean(fv):
    return float(sum(fv)) / len(fv)


def randomvariable(dividers):
    dividers = sorted(dividers)

    def fn(x):
        for i, d in enumerate(dividers):
            if x <= d:
                return i
        return len(dividers)

    return fn


def distribution(fv, rv, numclasses):
    seen = collections.defaultdict(lambda: 0)
    for f in fv:
        seen[rv(f)] += 1
    return lambda x: (seen[rv(x)] + 1.0) / (len(fv) + numclasses)


def model(fv):
    fv = np.array(fv)
    bins = np.linspace(min(fv), max(fv), 8)
    mean = fv.mean()
    digitized = np.digitize(fv, bins) - 1
    dividers = []
    for i in range(0, len(bins)):
        curr_split = fv[digitized == i]
        if not len(curr_split):
            dividers.append(mean)
        else:
            dividers.append(curr_split.mean())
    return distribution(fv, randomvariable(dividers), len(dividers) + 1)
