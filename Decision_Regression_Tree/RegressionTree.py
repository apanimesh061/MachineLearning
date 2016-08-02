from __future__ import division
from ReadData import Data
import numpy as np

depth = 0
depth_stack = []


def moving_average(splits):
    iterations = len(splits)
    new_splits = []
    for i in xrange(iterations - 1):
        if splits[i] < splits[i + 1]:
            new_val = (splits[i] + splits[i + 1]) / 2
            new_splits.append(new_val)
    return np.array(new_splits)


training_data = "Data\housing_train.txt"
test_data = "Data\housing_test.txt"
feature_list = ['CRIM',
                'ZN',
                'INDUS',
                'CHAS',
                'NOX',
                'RM',
                'AGE',
                'DIS',
                'RAD',
                'TAX',
                'PTRATIO',
                'B',
                'LSTAT']

dataset = Data(
        training_data,
        test_data,
        {index: feature_name for (index, feature_name) in enumerate(feature_list)})
a, b = dataset.get_data_set()
fnmap = dataset.get_feature_maps(normalized=False)
tfnmap = dataset.get_test_feature_maps()
feature_split_points = dataset.get_possible_thesh()
matrix, target = dataset.get_data_set()
test_matrix, test_target = dataset.get_test_set()
init_bit = range(len(target))
test_init_bit = range(len(test_target))
test_index_map = {index: inst for (index, inst) in enumerate(test_matrix.transpose())}


def update_features():
    temp_dict = dict()
    for (a, b) in feature_split_points.iteritems():
        temp_dict[a] = moving_average(b)
    return temp_dict


feature_split_ma = update_features()


class Node(object):
    def __init__(self, feature_index=None, threshold=None,
                 valid_bits=None, is_leaf=False,
                 left_child=None, right_child=None,
                 average=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.valid_bits = valid_bits
        self.is_leaf = is_leaf
        self.right = right_child
        self.left = left_child
        self.average = average


def regression_stats(valid_bits):
    if len(valid_bits) == 0:
        return {"target_mean": 0.0, "reduction": 0.0, "mse": 0.0}, 0
    target_meta = dict()
    sample_total = 0.0
    size_of_valid_bits = 0
    for index in valid_bits:
        size_of_valid_bits += 1
        sample_total += target[index]
    sample_mean = sample_total / size_of_valid_bits
    sample_sse = 0.0
    for index in valid_bits:
        sample_sse += (target[index] - sample_mean) ** 2
    sample_mse = sample_sse / size_of_valid_bits
    target_meta.update(
            {
                "target_mean": sample_mean,
                "reduction": sample_total ** 2 / size_of_valid_bits,
                "mse": sample_mse
            }
    )
    return target_meta, size_of_valid_bits


def split_set(feature_index, valid_rows, threshold):
    right_index = []
    left_index = []
    curr_feature = fnmap[feature_index]
    for valid_index in valid_rows:
        point = curr_feature[valid_index]
        if point <= threshold:
            left_index.append(valid_index)
        else:
            right_index.append(valid_index)
    return left_index, right_index


def grow_tree(valid_rows):
    global depth_stack
    depth_stack.append(1)
    if len(depth_stack) == 15:
        return Node(average=regression_stats(valid_rows)[0]['target_mean'], is_leaf=True)
    if len(valid_rows) < 6:
        return Node(average=regression_stats(valid_rows)[0]['target_mean'], is_leaf=True)
    optimal_exp = -1e309
    optimal_split_point = None
    optimal_splits = None
    for (index, feature_values) in feature_split_points.iteritems():
        possible_splits = feature_values
        for split_point in possible_splits:
            left_set, right_set = split_set(index, valid_rows, split_point)
            left_node_stats, left_node_size = regression_stats(left_set)
            right_node_stats, right_node_size = regression_stats(right_set)
            if not left_node_size or not right_node_size:
                continue
            sub_optimal_exp = left_node_stats["reduction"] + right_node_stats["reduction"]
            if sub_optimal_exp > optimal_exp:
                optimal_exp = sub_optimal_exp
                optimal_split_point = (index, split_point)
                optimal_splits = (left_set, right_set)

    new_node = Node(
            feature_index=optimal_split_point[0],
            threshold=optimal_split_point[1]
    )
    left_child = grow_tree(optimal_splits[0])
    depth_stack.pop()
    right_child = grow_tree(optimal_splits[1])
    depth_stack.pop()
    new_node.left = left_child
    new_node.right = right_child
    return new_node


def printer(tree, indent=' '):
    if tree.is_leaf:
        print str(tree.average)
    else:
        print str(dataset.feature_labels.get(tree.feature_index)) + ' < ' + str(tree.threshold) + ' ?'
        print indent + 'L -->',
        printer(tree.left, indent + '  ')
        print indent + 'R -->',
        printer(tree.right, indent + '  ')


def predictor(node, instance):
    while not node.is_leaf:
        value = instance[node.feature_index]
        threshold = node.threshold
        if value <= threshold:
            node = node.left
        else:
            node = node.right
    return node.average


ped = grow_tree(init_bit)
##printer(ped)

test_sse = 0.0
for (instance, val) in zip(test_matrix, test_target):
    reg = predictor(ped, instance)
    ##    print reg, val
    test_sse += (reg - val) ** 2

print test_sse / test_target.size

test_sse = 0.0
for (instance, val) in zip(matrix, target):
    reg = predictor(ped, instance)
    ##    print reg, val
    test_sse += (reg - val) ** 2

print test_sse / target.size
