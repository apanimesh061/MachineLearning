from __future__ import division
from adaboost.ActiveLearn import *

if __name__ == '__main__':
    proportion = 0.05

    data = Data(path="../Data/spambase.data")
    np.random.shuffle(data.data)
    split = round(len(data.data) * proportion)
    train = data.data[: split]
    test = data.data[split:]

    testing = to_data_point(test)
    training = to_data_point(train)

    print 'Testing count:', len(testing)
    print 'Training count:', len(training)
    print 'Feature count:', len(training[0].features)

    current_training_size = len(training) / data.no_of_instances
    while current_training_size <= 0.50:
        print "{:0.3f}% data is being trained...".format(current_training_size * 100.0)
        ada = AdaBoost(training, testing)
        ada.train()
        ada.classify()
        print
        training += ada.addition
        testing = ada.new_testing
        current_training_size = len(training) / data.no_of_instances
