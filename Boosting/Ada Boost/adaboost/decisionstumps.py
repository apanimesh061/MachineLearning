import math
import random
import itertools as it


class StumpLibrary(object):

    current_local_error = 0.0

    def __init__(self, dataset):
        self.sv = self.__class__.from_dataset(dataset)

    def pick_best(self, weights, foo=None):
        def optimality(s):
            err = Stump.error(s, weights)
            return [abs(0.5 - err), err]
        dist, err, i = max(optimality(s) + [i]
                           for i, s in it.imap(None, it.count(), self.sv))
        self.current_local_error = err
        return (self.sv.pop(i), dist) if foo else self.sv.pop(i)

    @classmethod
    def from_dataset(cls, dataset):
        stumps = []
        print "Loading stumps..."
        for i, fv in enumerate(it.izip(*(dp.features for dp in dataset))):
            for t in cls.thresholds(fv):
                s = i, t, Stump.mistakes((i, t, []), dataset)
                stumps.append(s)
        return stumps

    @staticmethod
    def thresholds(fv):
        fv = list(set(fv))
        fv.sort()
        # moving average
        tv = [(a + b) / 2.0 for a, b in it.imap(None, fv, fv[1:])]
        return [fv[0] - 1] + tv + [fv[-1] + 1]


class Stump(object):

    def __init__(self):
        raise NotImplementedError

    @classmethod
    def mistakes(cls, stump, dataset):
        return [i for i, dp in it.imap(None, it.count(), dataset) if cls.query(stump, dp) != dp.label]

    @staticmethod
    def query(stump, dp):
        i, t, _ = stump
        return 1 if dp.features[i] > t else -1

    @staticmethod
    def error(stump, weights):
        _, _, mistakes = stump
        return math.fsum(weights[m] for m in mistakes)
