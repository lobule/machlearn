from numpy import *
import operator


def classify(new_xs, factors, labels, k):
    differences = tile(new_xs, (factors.shape[0], 1)) - factors
    distance_ranks = (((differences**2).sum(1))**0.5).argsort()

    tally = {}
    for i in range(k):
        ith_label = labels[distance_ranks[i]]
        tally[ith_label] = tally.get(ith_label, 0) + 1

    return sorted(tally.items(), key=operator.itemgetter(1), reverse=True)[0][0]
