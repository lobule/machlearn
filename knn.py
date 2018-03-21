import numpy as np
import data


def classify(new_xs, factors, labels, k):
    differences = np.tile(new_xs, (factors.shape[0], 1)) - factors
    distance_ranks = (((differences**2).sum(1))**0.5).argsort()

    tally = {}
    for i in range(k):
        ith_label = labels[distance_ranks[i]]
        tally[ith_label] = tally.get(ith_label, 0) + 1

    return get_most_common(tally)


def get_most_common(tally):
    max_count = -1
    most_common_item = ""

    for item, count in tally.items():
        if count > max_count:
            most_common_item = item
            max_count = count

    return most_common_item


def normalize_cols(matrix):
    mins = matrix.min(0)
    maxes = matrix.max(0)
    ranges = maxes - mins

    rows = matrix.shape[0]
    normal_matrix = (matrix - np.tile(mins, (rows, 1))) / np.tile(ranges, (rows, 1))

    return normal_matrix, mins, ranges
