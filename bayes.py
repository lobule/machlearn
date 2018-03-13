from numpy import *

def get_token_set(corpus):
    tokens = set([])
    for sample in corpus:
        tokens = tokens | set(sample)
    return tokens


def get_vocabulary_vector(vocabulary, sample):
    vector = []
    for word in vocabulary:
        vector.extend([1 if word in sample else 0])
    return vector


def train_naive_bayes(factors, labels):
    n = len(factors)
    num_factors = len(factors[0])

    unique_labels = list(set(labels))
    num_labels = len(unique_labels)

    numerators = tile(0, (num_labels, num_factors))
    denominators = zeros(num_labels)
    ps = zeros(num_labels)

    for i in range(n):
        label_index = unique_labels.index(labels[i])

        numerators[label_index] += factors[i]
        denominators[label_index] += sum(factors[i])
        ps[label_index] += 1

    return numerators, denominators, ps / float(n), unique_labels
