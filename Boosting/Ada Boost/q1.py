from adaboost.Plotter import Plot
from adaboost.AdaBoost import *

if __name__ == '__main__':

    data = Data(path="../Data/spambase.data")
    np.random.shuffle(data.data)
    # split data to 10-90
    split = round(len(data.data) * 0.1)
    train = data.data[split:]
    test = data.data[:split]

    testing = to_data_point(test)
    training = to_data_point(train)

    print 'Testing count:', len(testing)
    print 'Training count:', len(training)
    print 'Feature count:', len(training[0].features)

    ada = AdaBoost(training, testing)
    ada.train()

    roc_pairs, auc_round_pairs, train_test_round_triplets, local_error_round_pairs = ada.classify()

    fp_rate = []
    tp_rate = []
    for fpr, tpr in rocdata(roc_pairs):
        fp_rate.append(fpr)
        tp_rate.append(tpr)

    plot = Plot()
    plot.roc(fp_rate, tp_rate, "spambase")
    plot.auc(auc_round_pairs)
    plot.error(train_test_round_triplets)
    plot.le(local_error_round_pairs)
