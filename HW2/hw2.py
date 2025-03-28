###### Your ID ######
# ID1: 322462920
# ID2: 207827825
#####################

import numpy as np
import matplotlib.pyplot as plt

### Chi square table values ###
# The first key is the degree of freedom
# The second key is the p-value cut-off
# The values are the chi-statistic that you need to use in the pruning

chi_table = {
    1: {0.5: 0.45, 0.25: 1.32, 0.1: 2.71, 0.05: 3.84, 0.0001: 100000},
    2: {0.5: 1.39, 0.25: 2.77, 0.1: 4.60, 0.05: 5.99, 0.0001: 100000},
    3: {0.5: 2.37, 0.25: 4.11, 0.1: 6.25, 0.05: 7.82, 0.0001: 100000},
    4: {0.5: 3.36, 0.25: 5.38, 0.1: 7.78, 0.05: 9.49, 0.0001: 100000},
    5: {0.5: 4.35, 0.25: 6.63, 0.1: 9.24, 0.05: 11.07, 0.0001: 100000},
    6: {0.5: 5.35, 0.25: 7.84, 0.1: 10.64, 0.05: 12.59, 0.0001: 100000},
    7: {0.5: 6.35, 0.25: 9.04, 0.1: 12.01, 0.05: 14.07, 0.0001: 100000},
    8: {0.5: 7.34, 0.25: 10.22, 0.1: 13.36, 0.05: 15.51, 0.0001: 100000},
    9: {0.5: 8.34, 0.25: 11.39, 0.1: 14.68, 0.05: 16.92, 0.0001: 100000},
    10: {0.5: 9.34, 0.25: 12.55, 0.1: 15.99, 0.05: 18.31, 0.0001: 100000},
    11: {0.5: 10.34, 0.25: 13.7, 0.1: 17.27, 0.05: 19.68, 0.0001: 100000}
}


def calc_gini(data):
    """
    Calculate gini impurity measure of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns:
    - gini: The gini impurity value.
    """
    gini = 0.0
    last_column = data[:, -1]
    _, counts = np.unique(last_column, return_counts=True)
    gini = 1 - np.sum((counts / len(last_column)) ** 2)
    return gini


def calc_entropy(data):
    """
    Calculate the entropy of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns:
    - entropy: The entropy value.
    """
    entropy = 0.0
    last_column = data[:, -1]
    _, counts = np.unique(last_column, return_counts=True)
    probabilities = counts / len(last_column)
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy


class DecisionNode:

    def __init__(self, data, impurity_func, feature=-1, depth=0, chi=1, max_depth=1000, gain_ratio=False):

        self.data = data  # the relevant data for the node
        self.feature = feature  # column index of criteria being tested
        self.pred = self.calc_node_pred()  # the prediction of the node
        self.depth = depth  # the current depth of the node
        self.children = []  # array that holds this nodes children
        self.children_values = []
        self.terminal = False  # determines if the node is a leaf
        self.chi = chi
        self.max_depth = max_depth  # the maximum allowed depth of the tree
        self.impurity_func = impurity_func
        self.gain_ratio = gain_ratio
        self.feature_importance = 0

    def calc_node_pred(self):
        """
        Calculate the node prediction.

        Returns:
        - pred: the prediction of the node
        """
        pred = None
        last_column = self.data[:, -1]
        classes, counts = np.unique(last_column, return_counts=True)
        pred = classes[np.argmax(counts)]
        return pred

    def add_child(self, node, val):
        """
        Adds a child node to self.children and updates self.children_values

        This function has no return value
        """
        self.children.append(node)
        self.children_values.append(val)

    def calc_feature_importance(self, n_total_sample):
        """
        Calculate the selected feature importance.

        Input:
        - n_total_sample: the number of samples in the dataset.

        This function has no return value - it stores the feature importance in 
        self.feature_importance
        """

        if self.feature is None:
            return

        self_size = len(self.data)
        prob = self_size / n_total_sample
        feature_column = self.data[:, self.feature]
        feature_values, counts = np.unique(feature_column, return_counts=True)
        feature_sum = 0
        for key, value in enumerate(feature_values):
            data_subset = self.data[feature_column == value]
            feature_sum += counts[key] / n_total_sample * self.impurity_func(data_subset)  # nopep8

        self.feature_importance = prob * self.impurity_func(self.data) - feature_sum  # nopep8

    def goodness_of_split(self, feature):
        """
        Calculate the goodness of split of a dataset given a feature and impurity function.

        Input:
        - feature: the feature index the split is being evaluated according to.

        Returns:
        - goodness: the goodness of split
        - groups: a dictionary holding the data after splitting according to the feature values.
        """

        if feature is None:
            return None, None

        goodness = 0
        groups = {}  # groups[feature_value] = data_subset
        feature_column = self.data[:, feature]
        feature_values = np.unique(feature_column)
        groups_impurity = 0
        split_info = 0

        if self.gain_ratio:
            impurity_func = calc_entropy
        else:
            impurity_func = self.impurity_func

        data_impurity = impurity_func(self.data)

        for value in feature_values:
            groups[value] = self.data[feature_column == value]
            group_prob = len(groups[value]) / len(self.data)
            groups_impurity += group_prob * impurity_func(groups[value])
            split_info -= group_prob * np.log2(group_prob)

        if self.gain_ratio:
            if split_info == 0:
                return 0, groups

            goodness = (data_impurity - groups_impurity) / split_info
        else:
            goodness = data_impurity - groups_impurity

        return goodness, groups

    def split(self):
        """
        Splits the current node according to the self.impurity_func. This function finds
        the best feature to split according to and create the corresponding children.
        This function should support pruning according to self.chi and self.max_depth.

        This function has no return value
        """
        if self.depth >= self.max_depth or self.impurity_func(self.data) == 0:
            self.terminal = True
            return

        best_feature = None
        best_goodness = 0
        best_groups = None

        for feature in range(self.data.shape[1] - 1):
            goodness, groups = self.goodness_of_split(feature)
            if goodness is not None and goodness > best_goodness:
                best_goodness = goodness
                best_feature = feature
                best_groups = groups

        self.feature = best_feature

        # chi pruning
        if best_groups is not None and self.feature is not None and len(best_groups) > 1 and check_chi(self, best_groups):
            for value, data_subset in best_groups.items():
                child = DecisionNode(data_subset, self.impurity_func, feature=self.feature, depth=self.depth + 1, chi=self.chi, max_depth=self.max_depth, gain_ratio=self.gain_ratio)  # nopep8
                self.add_child(child, value)
            return
        else:
            self.terminal = True
            return


def check_chi(node: DecisionNode, groups):

    if node.chi == 1:
        return True

    chi_squared_value = calc_chi(node, groups)
    degree_of_freedom = (len(groups) - 1) * (len(np.unique(node.data[:, -1])) - 1)  # nopep8
    return chi_squared_value > chi_table[degree_of_freedom][node.chi]


def calc_chi(node: DecisionNode, groups):
    chi_squared_value = 0
    unique_classes_values, unique_classes_count = np.unique(node.data[:, -1], return_counts=True)  # nopep8
    for value, data_subset in groups.items():
        subset_classes_values, subset_classes_count = np.unique(data_subset[:, -1], return_counts=True)  # nopep8
        for class_value_index, class_value in enumerate(unique_classes_values):
            expected = len(data_subset) * (unique_classes_count[class_value_index] / len(node.data))  # nopep8
            if class_value not in subset_classes_values:
                actual = 0
            else:
                actual = subset_classes_count[subset_classes_values.tolist().index(class_value)]  # nopep8
            chi_squared_value += ((actual - expected) ** 2) / expected

    return chi_squared_value


class DecisionTree:
    def __init__(self, data, impurity_func, feature=-1, chi=1, max_depth=1000, gain_ratio=False):
        self.data = data  # the relevant data for the tree
        self.impurity_func = impurity_func  # the impurity function to be used in the tree
        self.chi = chi
        self.max_depth = max_depth  # the maximum allowed depth of the tree
        self.gain_ratio = gain_ratio
        self.root = None  # the root node of the tree

    def build_tree(self):
        """
        Build a tree using the given impurity measure and training dataset. 
        You are required to fully grow the tree until all leaves are pure 
        or the goodness of split is 0.

        This function has no return value
        """
        self.root = DecisionNode(self.data, self.impurity_func, chi=self.chi, max_depth=self.max_depth, gain_ratio=self.gain_ratio)  # nopep8
        self.build_tree_recursive(self.root, n_total_sample=len(self.data))

    def build_tree_recursive(self, node: DecisionNode, n_total_sample):

        if node.terminal:
            return

        node.split()
        node.calc_feature_importance(n_total_sample)

        for child in node.children:
            self.build_tree_recursive(child, n_total_sample)

    def predict(self, instance: np.ndarray):
        """
        Predict a given instance

        Input:
        - instance: an row vector from the dataset. Note that the last element 
                    of this vector is the label of the instance.

        Output: the prediction of the instance.
        """
        pred = None
        stop = True  # initial value for the while loop
        node = self.root

        if node is None:
            return pred

        if node.terminal:
            return node.pred

        while not node.terminal and stop:
            stop = False
            feature_value = instance[node.feature]
            for i, child in enumerate(node.children):
                if node.children_values[i] == feature_value:
                    node = child
                    stop = True
                    break

        return node.pred

    def calc_accuracy(self, dataset):
        """
        Predict a given dataset

        Input:
        - dataset: the dataset on which the accuracy is evaluated

        Output: the accuracy of the decision tree on the given dataset (%).
        """
        accuracy = 0
        correct = 0
        for instance in dataset:
            if self.predict(instance) == instance[-1]:
                correct += 1
        accuracy = correct / len(dataset)
        return accuracy

    def depth(self):
        if self.root is None:
            return 0
        return self.root.depth


def depth_pruning(X_train, X_validation):
    """
    Calculate the training and validation accuracies for different depths
    using the best impurity function and the gain_ratio flag you got
    previously. On a single plot, draw the training and testing accuracy 
    as a function of the max_depth.

    Input:
    - X_train: the training data where the last column holds the labels
    - X_validation: the validation data where the last column holds the labels

    Output: the training and validation accuracies per max depth
    """
    training = []
    validation = []
    root = None
    for max_depth in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        root = DecisionTree(X_train, calc_entropy, max_depth=max_depth, gain_ratio=True)  # nopep8
        root.build_tree()
        training.append(root.calc_accuracy(X_train))
        validation.append(root.calc_accuracy(X_validation))
    return training, validation


def chi_pruning(X_train, X_test):
    """
    Calculate the training and validation accuracies for different chi values
    using the best impurity function and the gain_ratio flag you got
    previously. 

    Input:
    - X_train: the training data where the last column holds the labels
    - X_validation: the validation data where the last column holds the labels

    Output:
    - chi_training_acc: the training accuracy per chi value
    - chi_validation_acc: the validation accuracy per chi value
    - depth: the tree depth for each chi value
    """
    chi_training_acc = []
    chi_validation_acc = []
    depth = []
    for chi in [1, 0.5, 0.25, 0.1, 0.05, 0.0001]:
        tree = DecisionTree(X_train, calc_entropy, chi=chi, gain_ratio=True)
        tree.build_tree()
        chi_training_acc.append(tree.calc_accuracy(X_train))
        chi_validation_acc.append(tree.calc_accuracy(X_test))

        if tree.root is None:
            depth.append(0)
            continue

        tree_depth = calc_tree_depth(tree.root)
        depth.append(tree_depth)

    return chi_training_acc, chi_validation_acc, depth


def count_nodes(node):
    """
    Count the number of node in a given tree

    Input:
    - node: a node in the decision tree.

    Output: the number of node in the tree.
    """
    n_nodes = 1
    if node.children == []:
        return 1
    for child in node.children:
        n_nodes += count_nodes(child)
    return n_nodes


def calc_tree_depth(node: DecisionNode):
    """
    Calculate the depth of the tree.

    Input:
    - node: a node of type DecisionNode which is the root of the tree.

    Output: the depth of the tree.
    """

    depths = []

    # If the node is terminal - return the depth of the node.
    if node.terminal:
        return node.depth

    for node in node.children:
        node_depth = calc_tree_depth(node)
        depths.append(node_depth)

    return max(depths)  # return the maximum depth of the children
