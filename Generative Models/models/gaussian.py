import math


def mean(fv):
    return float(sum(fv)) / len(fv)


def mvue(fv, mu=None):
    mu = mean(fv) if mu is None else float(mu)
    return sum([math.pow(v - mu, 2) for v in fv]) / (len(fv) - 1)


def density(mu, var):
    if var <= 0.01:
        raise MinVarianceException
    sd = math.sqrt(var)
    a = 1 / (sd * math.sqrt(2 * math.pi))

    def pdf(x):
        b = -1 * math.pow(x - mu, 2) / (2 * var)
        return a * math.exp(b)
    return pdf


def model(fv):
    mu = mean(fv)
    var = mvue(fv)
    return density(mu, var)


class MinVarianceException(ValueError):
    pass
