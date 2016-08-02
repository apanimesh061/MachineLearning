class DataResult(object):
    def __init__(self, label_value, learner_value, predicted_label=None):
        self.label = label_value
        self.score = learner_value
        self.prediction = predicted_label


def applyop(operating_point, results):
    return [DataResult(dr.label, dr.score, int(dr.score > operating_point)) for dr in results]


def analyze(results):
    conf_mat = {'tp': 0.0,
            'fn': 0.0,
            'fp': 0.0,
            'tn': 0.0}

    for dr in results:
        conf_mat['tp'] += dr.label == 1 and dr.prediction == 1
        conf_mat['fn'] += dr.label == 1 and dr.prediction == 0
        conf_mat['fp'] += dr.label == 0 and dr.prediction == 1
        conf_mat['tn'] += dr.label == 0 and dr.prediction == 0

    try:
        conf_mat['fpr'] = conf_mat['fp'] / (conf_mat['fp'] + conf_mat['tn'])
    except ZeroDivisionError:
        conf_mat['fpr'] = 1.0
        print 'analyze ZeroDivisionError for fpr'

    try:
        conf_mat['fnr'] = conf_mat['fn'] / (conf_mat['tp'] + conf_mat['fn'])
    except ZeroDivisionError:
        conf_mat['fnr'] = 1.0
        print 'analyze ZeroDivisionError for fnr'

    try:
        conf_mat['tpr'] = conf_mat['tp'] / (conf_mat['fn'] + conf_mat['tp'])
    except ZeroDivisionError:
        conf_mat['tpr'] = 1.0
        print 'analyze ZeroDivisionError for tpr'

    conf_mat['oer'] = (conf_mat['fp'] + conf_mat['fn']) / len(results)

    return conf_mat


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
    conf_mat = analyze(applyop(results[0].score - 1, results))
    pairs = [(conf_mat['fpr'], conf_mat['tpr'])]
    for dr in results:
        conf_mat = analyze(applyop(dr.score, results))
        pairs.append((conf_mat['fpr'], conf_mat['tpr']))
    return pairs


def auc(r):
    r = sorted(r, reverse=True)
    return -0.5 * sum([(x1 - x0) * (y1 + y0) for (x0, y0), (x1, y1) in zip(r, r[1:])])
