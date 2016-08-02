class DataResult(object):
    def __init__(self, label_value, learner_value, predicted_label=None):
        self.label = label_value
        self.score = learner_value
        self.prediction = predicted_label


def applyop(operating_point, results):
    return [DataResult(dr.label, dr.score, int(dr.score > operating_point)) for dr in results]


def analyze(results):
    cmet = {'tp': 0.0,
            'fn': 0.0,
            'fp': 0.0,
            'tn': 0.0}

    for dr in results:
        cmet['tp'] += dr.label == 1 and dr.prediction == 1
        cmet['fn'] += dr.label == 1 and dr.prediction == 0
        cmet['fp'] += dr.label == 0 and dr.prediction == 1
        cmet['tn'] += dr.label == 0 and dr.prediction == 0

    try:
        cmet['fpr'] = cmet['fp'] / (cmet['fp'] + cmet['tn'])
    except ZeroDivisionError:
        cmet['fpr'] = 1.0
        print 'analyze ZeroDivisionError for fpr'

    try:
        cmet['fnr'] = cmet['fn'] / (cmet['tp'] + cmet['fn'])
    except ZeroDivisionError:
        cmet['fnr'] = 1.0
        print 'analyze ZeroDivisionError for fnr'

    try:
        cmet['tpr'] = cmet['tp'] / (cmet['fn'] + cmet['tp'])
    except ZeroDivisionError:
        cmet['tpr'] = 1.0
        print 'analyze ZeroDivisionError for tpr'

    cmet['oer'] = (cmet['fp'] + cmet['fn']) / len(results)

    return cmet


def minerrop(results):
    op = 0.0
    et = analyze(applyop(op, results))
    direction = 0.1 * (1 if et['fpr'] > et['fnr'] else -1)
    preverr = et['oer']
    while True:
        op += direction
        et = analyze(applyop(op, results))
        if et['oer'] > preverr:
            return op - direction
        else:
            preverr = et['oer']


def rocdata(results):
    results = sorted(results, key=lambda x: x.score)
    cmet = analyze(applyop(results[0].score - 1, results))
    pairs = [(cmet['fpr'], cmet['tpr'])]
    for dr in results:
        cmet = analyze(applyop(dr.score, results))
        pairs.append((cmet['fpr'], cmet['tpr']))
    return pairs


def auc(r):
    r = sorted(r, reverse=True)
    return -0.5 * sum([(x1 - x0) * (y1 + y0) for (x0, y0), (x1, y1) in zip(r, r[1:])])
