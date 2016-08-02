from __future__ import division
import numpy as np
from scipy import spatial
np.random.seed(123)  # 10
np.set_printoptions(suppress=True)


class Data(object):
    data = None
    X = None
    y = None
    no_of_instances = None
    vocab_size = None
    fold_no = None

    def __init__(self, path, fold_no=0):
        self.data = np.loadtxt(path, delimiter=',', dtype=np.float32)
        self.X = self.data[:, range(self.data.shape[1] - 1)]
        self.y = self.data[:, [-1]]
        self.no_of_instances, self.vocab_size = self.X.shape
        self.fold_no = fold_no

    def generate_k_folds(self, K=10, randomise=True):
        matrix = [i for i in xrange(len(self.y))]
        if randomise:
            from random import shuffle
            matrix = list(matrix)
            shuffle(matrix)
        for k in xrange(K):
            training = [x for i, x in enumerate(matrix) if i % K != k]
            validation = [x for i, x in enumerate(matrix) if i % K == k]
            yield np.array(training), np.array(validation)


class Kernels(object):
    @staticmethod
    def euclidian(input_vector, prototype_vector):
        return np.linalg.norm(input_vector - prototype_vector)

    @staticmethod
    def cosine(input_vector, prototype_vector):
        return spatial.distance.cosine(input_vector, prototype_vector)

    @staticmethod
    def gaussian(input_vector, prototype_vector):
        # gamma = 10e-8
        gamma = 2.0
        dist = np.linalg.norm(input_vector - prototype_vector)
        dist **= 2
        return np.exp(-1.0 * gamma * dist)

    @staticmethod
    def polynomial(input_vector, prototype_vector):
        lam = 0.002
        return (lam * np.dot(input_vector, prototype_vector) + 30) ** 2


class KernelDensity(object):
    def __init__(self, train, test, classes):
        self.train = train
        self.test = test
        self.classes = classes

    def run(self, metric):
        closeness_metric = None
        if metric == "gaussian":
            closeness_metric = Kernels.gaussian
        elif metric == "euclidian":
            closeness_metric = Kernels.euclidian
        elif metric == "cosine":
            closeness_metric = Kernels.cosine
        elif metric == "poly":
            closeness_metric = Kernels.polynomial
        else:
            raise NotImplementedError

        class_wise = dict()
        class_prior = dict()
        for klass in self.classes:
            class_wise[klass] = self.train[self.train[:, -1] == klass][:, range(self.train.shape[1] - 1)]
            class_prior[klass] = class_wise[klass].shape[0] / self.train.shape[0]

        count = 0
        for test_vector in self.test:
            prototype_label = int(test_vector[-1])
            prototype_feature = test_vector[: -1]
            class_posterior = dict()
            class_probability = dict()

            for klass in self.classes:
                kernel_sum = 0.0
                for class_instance in class_wise[klass]:
                    kernel_sum += closeness_metric(input_vector=class_instance, prototype_vector=prototype_feature)
                mc = class_wise[klass].shape[0]
                class_posterior[klass] = (1 / mc) * kernel_sum
                class_probability[klass] = class_prior[klass] * class_posterior[klass]

            best_class = sorted(class_probability.iteritems(), key=lambda k: k[1], reverse=True)[0][0]

            if best_class == prototype_label:
                count += 1

        return (count / self.test.shape[0]) * 100.0


class Relief(object):
    relevance_vector = None
    good_features = None

    def __init__(self, train, classes, epochs):
        self.train = train
        self.m = epochs
        self.targets = self.train[:, -1]
        self.feature_data = self.train[:, range(self.train.shape[1] - 1)]
        self.classes = classes

    def apply(self):
        n, p = self.feature_data.shape
        w = np.zeros(p, dtype=np.float32)
        iters = np.asarray(range(n))
        np.random.shuffle(iters)

        class_wise = dict()
        class_wise_prior = dict()
        for klass in self.classes:
            class_wise[klass] = self.train[self.train[:, -1] == klass]
            class_wise_prior[klass] = class_wise[klass].shape[0] / self.train.shape[0]

        for x in iters[:self.m]:
            current_instance = self.train[x]
            feature_vec = current_instance[:-1]
            feature_label = current_instance[-1]
            nearest = dict()
            for klass in self.classes:
                best = np.inf
                best_instance = None
                for i in class_wise[klass]:
                    current_dist = Kernels.euclidian(i[:-1], feature_vec)
                    if current_dist < best and current_dist != 0.0:
                        best = current_dist
                        best_instance = i[:-1]
                nearest[klass] = best_instance
            near_hit = nearest[feature_label]
            near_miss = nearest[self.classes[self.classes != feature_label][0]]
            w = np.subtract(w, np.subtract(np.square(np.subtract(feature_vec, near_hit)),
                                           np.square(np.subtract(feature_vec, near_miss))))
            self.relevance_vector = w / self.m
            self._top_features()

    def _top_features(self, n=5):
        rel_vec = self.relevance_vector.copy()
        a = np.c_[rel_vec, np.arange(0, rel_vec.size, 1)]
        self.good_features = a[a[:, 0].argsort()][::-1][:n][:, 1].astype(np.uint)

    def create_new_data(self):
        return np.c_[self.feature_data[:, self.good_features], self.targets]


class KNN(object):
    train = None
    test = None
    window = None

    def __init__(self, train, test, k=None):
        self.train = train
        self.test = test
        self.X = train[:, range(self.train.shape[1] - 1)]
        self.y = train[:, -1]
        self.k = k

    def set_parzen_window(self, window):
        self.window = window

    def classify(self, metric):
        def __majority_voting(label_array):
            label, pos = np.unique(label_array, return_inverse=True)
            counts = np.bincount(pos)
            if counts.size > 1:
                if not np.bitwise_xor.reduce(counts):
                    return __majority_voting(label_array[:-1])
            maxpos = counts.argmax()
            return label[maxpos]

        closeness_metric = None
        if metric == "gaussian":
            closeness_metric = Kernels.gaussian
        elif metric == "euclidian":
            closeness_metric = Kernels.euclidian
        elif metric == "cosine":
            closeness_metric = Kernels.cosine
        elif metric == "poly":
            closeness_metric = Kernels.polynomial
        else:
            raise NotImplementedError

        if self.window:
            count = 0
        else:
            count = np.array([0] * len(self.k))
        for test_feature in self.test:
            prototype_feature = test_feature[:-1]
            proto_label = int(test_feature[-1])
            similarity_matrix = np.array([], dtype=np.float32)
            for index, train_feature in enumerate(self.train):
                input_feature = train_feature[:-1]
                input_label = train_feature[-1]
                closeness = closeness_metric(input_vector=input_feature, prototype_vector=prototype_feature)
                sim = np.array([closeness, input_label])
                if index == 0:
                    similarity_matrix = np.hstack((similarity_matrix, sim))
                else:
                    similarity_matrix = np.vstack((similarity_matrix, sim))

            a = similarity_matrix
            if metric is "gaussian" or metric is "poly":
                a = a[a[:, 0].argsort()][::-1]
            else:
                a = a[a[:, 0].argsort()]

            if self.window:
                label_list = a[a[:, 0] <= self.window][:, 1]
                if len(label_list) == 0:
                    label_list = np.random.randint(0, np.unique(self.y).size, 1)
                predicted_label = int(__majority_voting(label_list))
                if predicted_label == proto_label:
                    count += 1
            else:
                for idx, k in enumerate(self.k):
                    predicted_label = int(__majority_voting(a[:k][:, -1]))
                    if predicted_label == proto_label:
                        count[idx] += 1
            del a
        return (count / self.test.shape[0]) * 100.0
