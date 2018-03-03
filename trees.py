from math import log
import operator


def calculate_entropy(data):
    n = len(data)
    label_tally = {}
    for datum in data:
        label = datum if not isinstance(datum, list) else datum[-1]
        if label not in label_tally.keys():
            label_tally[label] = 0
        label_tally[label] += 1
    entropy = 0.0
    for label in label_tally:
        p = label_tally[label] / float(n)
        entropy -= p * log(p, 2)
    return entropy


def split(matrix, feature_index, feature_value):
    sub_matrix = []
    for row in matrix:
        if row[feature_index] == feature_value:
            sub_matrix.append(row[:feature_index] + row[feature_index + 1:])
    return sub_matrix


def find_optimal_split_feature(matrix):
    num_features = len(matrix[0]) - 1
    entropy = calculate_entropy(matrix)
    max_information_gain = 0
    optimal_feature = -1

    for i in range(num_features):
        values = set([row[i] for row in matrix])
        new_entropy = 0
        for value in values:
            sub_matrix = split(matrix, i, value)
            p = len(sub_matrix) / float(len(matrix))
            new_entropy += p * calculate_entropy(sub_matrix)
        information_gain = entropy - new_entropy
        if information_gain > max_information_gain:
            max_information_gain = information_gain
            optimal_feature = i

    return optimal_feature


def get_majority(data):
    tally = {}
    for item in data:
        if item not in tally.keys():
            tally[item] = 0
        tally[item] += 1
    return sorted(tally.items(), key=operator.itemgetter(1), reverse=True)[0][0]


def create_tree(matrix, labels):
    ys = [row[-1] for row in matrix]
    if ys.count(ys[0]) == len(ys):
        return ys[0]

    if len(matrix[0]) == 1:
        return get_majority(ys)

    optimal_feature = find_optimal_split_feature(matrix)
    optimal_feature_label = labels[optimal_feature]

    tree = {optimal_feature_label: {}}
    labels_copy = labels[:]
    del(labels_copy[optimal_feature])

    values = set([row[optimal_feature] for row in matrix])
    for value in values:
        tree[optimal_feature_label][value] = create_tree(split(matrix, optimal_feature, value), labels_copy)

    return tree


def classify(new_xs, tree, labels):
    first_label = list(tree.keys())[0]
    subtree = tree[first_label]
    label_index = labels.index(first_label)

    for key in subtree.keys():
        if new_xs[label_index] == key:
            if type(subtree[key]).__name__ == 'dict':
                return classify(new_xs, subtree[key], labels)
            else:
                return subtree[key]


def save_tree(tree, filename):
    import pickle
    writer = open(filename, 'wb')
    pickle.dump(tree, writer)
    writer.close()


def load_tree(filename):
    import pickle
    reader = open(filename, 'rb')
    return pickle.load(reader)

