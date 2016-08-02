from __future__ import division
import math
import collections
import numpy as np
import matplotlib.pyplot as plt
from numpy import trapz
from models import bernoulli, histogram, histogram_9bin, gaussian
from tabulate import tabulate

np.errstate(divide='ignore')


class Data(object):
    data = None
    X = None
    y = None
    no_of_instances = None
    vocab_size = None

    def __init__(self, path='../Data/spambase/spambase.data'):
        self.data = np.loadtxt(path, delimiter=',', dtype='float')
        self.X = self.data[:, range(self.data.shape[1] - 1)]
        self.y = self.data[:, [-1]]
        self.no_of_instances, self.vocab_size = self.X.shape

    def generate_k_folds(self, K=10, randomise=True):
        matrix = self.X
        target = self.y
        matrix = [i for i in xrange(len(target))]
        if randomise:
            from random import shuffle
            matrix = list(matrix)
            shuffle(matrix)
        for k in xrange(K):
            training = [x for i, x in enumerate(matrix) if i % K != k]
            validation = [x for i, x in enumerate(matrix) if i % K == k]
            yield training, validation


def naivebayes(testing, training, model):
    T = []
    F = []
    for dp in training:
        (T if dp['label'] else F).append(dp['features'])
    Tprior = float(len(T)) / len(training)
    Fprior = float(len(F)) / len(training)
    priorlogodds = math.log(Tprior / Fprior)
    Tm = []
    Fm = []
    for Tfv, Ffv in zip(zip(*T), zip(*F)):
        try:
            tm = model(Tfv)
            fm = model(Ffv)
        except gaussian.MinVarianceException:
            tm = lambda x: 0.01
            fm = lambda x: 0.01

        Tm.append(tm)
        Fm.append(fm)

    ret = [dp.copy() for dp in testing]
    for dp in ret:
        dp['score'] = priorlogodds
        for idx, val in enumerate(dp['features']):
            gT = Tm[idx](val)
            gF = Fm[idx](val)
            if gT and gF:
                dp['score'] += math.log(gT / gF)
        del dp['features']
    return ret


def argmax(augmented):
    """
    REF: http://www0.cs.ucl.ac.uk/staff/ucacbbl/roc/
    """
    p = 0.0
    for val in augmented:
        if val['label'] == 1:
            p += 1
    p /= len(augmented)  # proportion of positives

    op = 0.0
    res = get_confusion_matrix(set_labels(op, augmented))
    gradient = (res['fpr'] / res['fnr']) * ((1 - p) / p)
    prev_accuracy = res['accuracy']
    while True:
        op += gradient
        res = get_confusion_matrix(set_labels(op, augmented))
        if res['accuracy'] < prev_accuracy:
            return op - gradient
        else:
            prev_accuracy = res['accuracy']


def set_labels(op, augmented):
    for dp in augmented:
        dp['prediction'] = int(dp['score'] > op)
    return augmented


def get_confusion_matrix(predicted):
    """
    Calculates accuracy on the basis of caluculated
    label and predicted label
    """
    results = {'tp': 0.0, 'fn': 0.0, 'fp': 0.0, 'tn': 0.0}
    for dp in predicted:
        results['tp'] += dp['label'] == 1 and dp['prediction'] == 1
        results['fn'] += dp['label'] == 1 and dp['prediction'] == 0
        results['fp'] += dp['label'] == 0 and dp['prediction'] == 1
        results['tn'] += dp['label'] == 0 and dp['prediction'] == 0
    results['fpr'] = results['fp'] / (results['fp'] + results['tn'])
    results['fnr'] = results['fn'] / (results['tp'] + results['fn'])
    results['tpr'] = results['tp'] / (results['fn'] + results['tp'])
    results['accuracy'] = 1 - ((results['fp'] + results['fn']) / len(predicted))
    return results


def rocdata(results):
    results['dpresults'] = sorted(results['dpresults'], key=lambda x: x['score'])
    r = get_confusion_matrix(set_labels(results['dpresults'][0]['score'] - 1, results['dpresults'][:]))
    pairs = [(r['fpr'], r['tpr'])]
    for dp in results['dpresults']:
        r = get_confusion_matrix(set_labels(dp['score'], results['dpresults'][:]))
        pairs.append((r['fpr'], r['tpr']))
    return pairs


def create_dict(dataset, labels):
    new_set = []
    for (i, j) in zip(dataset, labels):
        new_set.append({"features": i, "label": int(j[0])})
    return new_set


def plot(plot_list, roc_types):
    aucs = []
    for (x, y), _ in zip(plot_list, roc_types):
        aucs.append(-trapz(y, x))
        plt.plot(x, y)

    legend = []
    for auc, name in zip(aucs, roc_types):
        legend.append("{0} NB with AUC = {1:0.3f}".format(name.capitalize(), auc))

    plt.legend(legend, loc='lower right')
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.grid()
    plt.show()


if __name__ == '__main__':
    data = Data()
    FOLDCOUNT = 10
    ROCFOLD = 0
    results = collections.defaultdict(list)

    for name, model in [(m.__name__, m.model) for m in [bernoulli]]:  # , gaussian, histogram, histogram_9bin]]:
        print "Computing for {0}".format(name.capitalize())
        table = []
        for k, (current_train_indices, current_test_indices) in enumerate(data.generate_k_folds()):
            current_training_set = data.X[current_train_indices].tolist()
            current_training_target = data.y[current_train_indices].tolist()
            current_testing_set = data.X[current_test_indices].tolist()
            current_testing_target = data.y[current_test_indices].tolist()
            training = create_dict(current_training_set, current_training_target)
            testing = create_dict(current_testing_set, current_testing_target)

            augmented = naivebayes(testing, training, model)
            op = argmax(augmented)
            augmented = set_labels(op, augmented)
            r = get_confusion_matrix(augmented)
            r.update({"current_fold": k, "dpresults": augmented})
            results[name].append(r)

        for r in results[name]:
            table.append([r['current_fold'], r['fpr'], r['fnr'], r['accuracy'], 1 - r['accuracy']])
        table.append(['Average',
                      sum([r['fpr'] for r in results[name]]) / FOLDCOUNT,
                      sum([r['fnr'] for r in results[name]]) / FOLDCOUNT,
                      sum([r['accuracy'] for r in results[name]]) / FOLDCOUNT,
                      sum([1 - r['accuracy'] for r in results[name]]) / FOLDCOUNT])
        print tabulate(table, headers=["Current Fold", "FP Rate", "FN Rate", "Accuracy", "Error Rate"], tablefmt="psql")
        print

    plot_list = []
    roc_types = []
    for name, r in [(k, v[ROCFOLD]) for k, v in results.iteritems()]:
        roc_data = zip(*rocdata(r))
        x, y = np.array(list(roc_data[0])), np.array(list(roc_data[1]))
        plot_list.append((x, y))
        roc_types.append(name)
    plot(plot_list, roc_types)
