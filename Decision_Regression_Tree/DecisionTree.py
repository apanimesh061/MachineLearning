from __future__ import division
from ReadData import KFoldData
from collections import defaultdict, Counter
import itertools
from math import log

depth = 0
depth_stack = []
max_folds = 10


def mean_shift_normalisation(arr):
    mu_mat = arr.mean(axis=0)
    std_mat = arr.std(axis=0)
    arr = (arr - mu_mat) / std_mat  # normalize by zero mean and unit variance
    return arr


training_data = "Data\spambase\spambase.data"
feature_list = ['word_freq_make', 'word_freq_address', 'word_freq_all',
                'word_freq_3d', 'word_freq_our', 'word_freq_over', 'word_freq_remove',
                'word_freq_internet', 'word_freq_order', 'word_freq_mail', 'word_freq_receive',
                'word_freq_will', 'word_freq_people', 'word_freq_report',
                'word_freq_addresses', 'word_freq_free', 'word_freq_business',
                'word_freq_email', 'word_freq_you', 'word_freq_credit', 'word_freq_your',
                'word_freq_font', 'word_freq_000', 'word_freq_money', 'word_freq_hp',
                'word_freq_hpl', 'word_freq_george', 'word_freq_650', 'word_freq_lab',
                'word_freq_labs', 'word_freq_telnet', 'word_freq_857', 'word_freq_data',
                'word_freq_415', 'word_freq_85', 'word_freq_technology', 'word_freq_1999',
                'word_freq_parts', 'word_freq_pm', 'word_freq_direct', 'word_freq_cs',
                'word_freq_meeting', 'word_freq_original', 'word_freq_project', 'word_freq_re',
                'word_freq_edu', 'word_freq_table', 'word_freq_conference', 'char_freq_;',
                'char_freq_(', 'char_freq_[', 'char_freq_!', 'char_freq_$', 'char_freq_#',
                'capital_run_length_average', 'capital_run_length_longest',
                'capital_run_length_total']

dataset = KFoldData(training_data, {index: feature_name for (index, feature_name) in enumerate(feature_list)})
dataset.get_data_set(True)
fnmap = dataset.get_feature_maps()
feature_split_points = dataset.get_possible_ranges()
matrix = dataset.full_data
target = dataset.target
instance_index_map = {index: inst for (index, inst) in enumerate(matrix)}


class Node(object):
    def __init__(self, feature_index=None, threshold=None,
                 valid_bits=None, is_leaf=False,
                 left_child=None, right_child=None,
                 max_freq=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.valid_bits = valid_bits
        self.is_leaf = is_leaf
        self.right = left_child
        self.left = right_child
        self.max_freq = max_freq


def decision_stats(valid_bits):
    target_meta = dict()
    class_count = defaultdict(int)
    size_of_valid_bits = 0
    for index in valid_bits:
        size_of_valid_bits += 1
        val = target[index]
        class_count[val] += 1
    entropy = 0.0
    for (label, count) in class_count.iteritems():
        curr_prob = count / size_of_valid_bits
        entropy += -1.0 * curr_prob * log(curr_prob, 2)
    target_meta.update(
            {
                "target_classes": class_count,
                "target_entropy": entropy
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


def moving_average(splits):
    splits = sorted(splits)
    new_splits = []
    for i in xrange(len(splits) - 1):
        if splits[i] < splits[i + 1]:
            new_val = (splits[i] + splits[i + 1]) / 2
            new_splits.append(new_val)
    return new_splits


def grow_tree(valid_rows):
    global depth_stack
    depth_stack.append(1)
    if len(depth_stack) == 15 or len(valid_rows) < 50:
        klasses = decision_stats(valid_rows)[0]["target_classes"]
        return Node(max_freq=sorted(klasses.items(), key=lambda x: x[1], reverse=True)[0][0], is_leaf=True)
    parent_node_stats, parent_node_size = decision_stats(valid_rows)
    optimal_exp = -1e309
    optimal_split_point = None
    optimal_splits = None
    for (index, feature_values) in feature_split_points.iteritems():
        possible_splits = feature_values
        for split_point in possible_splits:
            left_set, right_set = split_set(index, valid_rows, split_point)
            left_node_stats, left_node_size = decision_stats(left_set)
            right_node_stats, right_node_size = decision_stats(right_set)
            if not left_node_size or not right_node_size:
                continue
            sub_optimal_exp = parent_node_stats["target_entropy"] - \
                              (
                                  (left_node_size / parent_node_size) * left_node_stats["target_entropy"] +
                                  (right_node_size / parent_node_size) * right_node_stats["target_entropy"]
                              )
            if sub_optimal_exp > optimal_exp:
                optimal_exp = sub_optimal_exp
                optimal_split_point = (index, split_point)
                optimal_splits = (left_set, right_set)
    print "@ depth", len(depth_stack)
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
        print str(tree.max_freq)
    else:
        print str(dataset.feature_labels[tree.feature_index]) + ' < ' + str(tree.threshold) + ' ?'
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
    return node.max_freq


final_accuracy = 0.0
fold_count = 1
for train_set, test_set in dataset.k_fold_cross_validation(K=max_folds, randomise=True):
    print "Training fold", fold_count, "..."
    ped = grow_tree(train_set)  # training
    printer(ped)
    correct_count = 0
    print "Testing fold", fold_count, "..."
    asdf = []
    conf_mat = defaultdict(int)
    for index in test_set:  # testing
        current_instance = matrix[index]
        current_target = target[index]
        label = predictor(ped, current_instance)
        asdf.append((label, current_target))
        if label == 1 and current_target == 1:
            conf_mat['TP'] += 1
        elif label == 0 and current_target == 0:
            conf_mat['TN'] += 1
        elif label == 1 and current_target == 0:
            conf_mat['FP'] += 1
        elif label == 0 and current_target == 1:
            conf_mat['FN'] += 1
        else:
            pass
        if label == current_target:
            correct_count += 1
    current_acc = correct_count / len(test_set)
    print "Accuracy for fold {0} is {1} %".format(fold_count, current_acc * 100.0)
    print asdf
    print conf_mat
    final_accuracy += current_acc

    correct_count = 0
    print "Testing train fold", fold_count, "..."
    for index in train_set:  # testing
        current_instance = matrix[index]
        current_target = target[index]
        label = predictor(ped, current_instance)
        if label == current_target: correct_count += 1
    current_acc = correct_count / len(train_set)
    print "Accuracy for training fold {0} is {1} %".format(fold_count, current_acc * 100.0)
    final_accuracy += current_acc

    fold_count += 1
    exit()

print "Average accuracy is {0} %".format((final_accuracy / max_folds) * 100.0)
