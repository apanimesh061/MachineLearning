import numpy as np
import bitarray


class Data(object):
    def __init__(self, path_to_training, path_to_test, feature_labels):
        self.feature_labels = feature_labels
        self.train = path_to_training
        self.test = path_to_test

    def _shift_scale(self, arr):
        min_of_arr = np.min(arr)
        arr = map(lambda v: v - min_of_arr, arr)
        max_of_new_arr = np.max(arr)
        return map(lambda v: v / max_of_new_arr, arr)

    def get_feature_labels(self):
        return self.feature_labels

    def get_data_set(self, normalize=False):
        training_matrix = []
        target_data = []
        with open(self.train, 'rb') as train_file_obj:
            data = train_file_obj.readlines()
            for line in data:
                line = line.strip()
                if line:
                    line = line.split()
                    if normalize:
                        training_matrix.append(self._shift_scale([float(a) for a in line[:-1]]))
                    else:
                        training_matrix.append([float(a) for a in line[:-1]])
                    target_data.append(float(line[-1]))
        if normalize:
            normalized_target = self._shift_scale(np.asarray(target_data, dtype='float32'))
            return np.asarray(training_matrix, dtype='float32'), normalized_target
        else:
            return np.asarray(training_matrix, dtype='float32'), \
                   np.asarray(target_data, dtype='float32')

    def get_feature_maps(self, normalized=False):
        a, _ = self.get_data_set()
        return dict(zip(self.feature_labels, a.transpose()))

    def get_possible_thesh(self):
        a, _ = self.get_data_set()
        fin_dict = dict()
        for index, row in enumerate(a.transpose()):
            fin_dict.update({index: np.asarray(sorted(list(set(row))))})
        return fin_dict

    def get_matrix_generator(self, matrix):
        for row in matrix: yield row

    def get_test_set(self, normalize=False):
        training_matrix = []
        target_data = []
        with open(self.test, 'rb') as train_file_obj:
            data = train_file_obj.readlines()
            for line in data:
                line = line.strip()
                if line:
                    line = line.split()
                    if normalize:
                        training_matrix.append(self._shift_scale([float(a) for a in line[:-1]]))
                    else:
                        training_matrix.append([float(a) for a in line[:-1]])
                    target_data.append(float(line[-1]))
        if normalize:
            normalized_target = self._shift_scale(np.asarray(target_data, dtype='float32'))
            return np.asarray(training_matrix, dtype='float32'), \
                   np.asarray(normalized_target, dtype='float32')
        else:
            return np.asarray(training_matrix, dtype='float32'), \
                   np.asarray(target_data, dtype='float32')

    def get_test_feature_maps(self):
        _, a = self.get_test_set()
        return dict(zip(self.feature_labels, a.transpose()))


class KFoldData(object):
    def __init__(self, path_to_training, feature_labels):
        self.feature_labels = feature_labels
        self.train = path_to_training
        self.target = None
        self.full_data = None

    def get_feature_labels(self):
        return self.feature_labels

    def _shift_scale(self, arr):
        min_of_arr = np.min(arr)
        arr = map(lambda v: v - min_of_arr, arr)
        max_of_new_arr = np.max(arr)
        return map(lambda v: v / max_of_new_arr, arr)

    def get_target(self):
        return self.target

    def get_full_data(self):
        return self.full_data

    def get_data_set(self, normalize=False):
        with open(self.train) as f:
            data = f.readlines()
            training_matrix = []
            target_data = []
            for line in data:
                line = line.strip()
                line = line.split(',')
                if line:
                    if normalize:
                        training_matrix.append(self._shift_scale([float(a) for a in line[:-1]]))
                    else:
                        training_matrix.append([float(a) for a in line[:-1]])
                    target_data.append(float(line[-1]))
        if normalize:
            normalized_target = self._shift_scale(np.asarray(target_data, dtype='float32'))
            self.target = np.asarray(normalized_target, dtype='int')
        else:
            self.target = np.asarray(target_data, dtype='int')
        self.full_data = np.asarray(training_matrix, dtype='float32')

    def get_feature_maps(self):
        return dict(zip(self.feature_labels, self.full_data.transpose()))

    def get_possible_thresh(self):
        fin_dict = dict()
        for index, row in enumerate(self.full_data.transpose()):
            fin_dict.update({index: np.asarray(sorted(list(set(row))))})
        return fin_dict

    def get_possible_ranges(self):
        fin_dict = dict()
        for index, row in enumerate(self.full_data.transpose()):
            temp_list = np.asarray(sorted(list(set(row))))
            range_list = self._moving_average_4(temp_list)
            fin_dict.update({index: range_list})
        return fin_dict

    def _moving_average_4(self, splits):
        splits = sorted(splits)
        iterations = len(splits)
        new_splits = []
        for i in xrange(iterations):
            if i + 20 < iterations:
                if splits[i] < splits[i + 20]:
                    new_val = (splits[i] + splits[i + 20]) / 2
                    new_splits.append(new_val)
        return np.asarray(splits)

    def get_matrix_generator(self, matrix):
        for row in matrix: yield row

    def k_fold_cross_validation(self, K=10, randomise=False):
        X = [i for i in xrange(len(self.target))]
        if randomise:
            from random import shuffle
            X = list(X)
            shuffle(X)
        for k in xrange(K):
            training = [x for i, x in enumerate(X) if i % K != k]
            validation = [x for i, x in enumerate(X) if i % K == k]
            yield training, validation

    def list_to_bitarray(self, some_list):
        b = bitarray.bitarray([False] * len(self.target))
        for no in some_list: b[no] = True
        return b

##with open("Data\spambase\spambase.names", "rb") as f:
##    data = f.readlines()
##    for line in data:
##        line = line.strip()
##        if not line.startswith("|") and line and "|" not in line:
##            feature_name, feature_type = line.split(':')
##            feature_names.append(feature_name.strip())
