import math
import itertools as it


class Boosting(object):

    current_score = None

    def __init__(self, dataset):
        self.data = dataset
        # uniform weight
        self.weights = len(dataset) * [1.0 / len(dataset)]
        self._fn = []
        self._loop = self.__loop()

    @property
    def weighted_data(self):
        return it.imap(None, self.weights, self.data)

    def init(self):
        return self._loop.next()

    def round(self, c):
        return self._loop.send(c)

    def __loop(self):
        while True:
            classify = yield self.weights[:]
            error = math.fsum(w for w, dp in self.weighted_data if classify(dp) != dp.label)
            assert 0 <= error <= 1
            alpha = 0.5 * math.log((1 - error) / error)
            self.weights = [w * math.e ** (-alpha * classify(dp) * dp.label) for w, dp in self.weighted_data]
            z = math.fsum(self.weights)
            self.weights = [w / z for w in self.weights]
            curried = (lambda cls, c: lambda dp: cls(dp) * c)(classify, alpha)
            self._fn.append(curried)

    def model(self, dp):
        return math.fsum(fn(dp) for fn in self._fn)

    def hypothesis(self, dp):
        val = self.model(dp)
        return 1 if val > 0 else -1
