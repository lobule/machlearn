import numpy as np


def get_token_set(corpus):
    tokens = set([])
    for sample in corpus:
        tokens = tokens | set(sample)
    return tokens


def get_vocabulary_set_vector(vocabulary, sample):
    vector = []
    for word in vocabulary:
        vector.extend([1 if word in sample else 0])
    return vector


def get_vocabulary_bag_vector(vocabulary, sample):
    vector = np.zeros(len(vocabulary))
    for word in sample:
        if word in vocabulary:
            vector[vocabulary.index(word)] += 1
    return vector


def train_naive_bayes(factors, labels):
    n = len(factors)
    num_factors = len(factors[0])

    unique_labels = list(set(labels))
    num_labels = len(unique_labels)

    numerators = np.tile(0, (num_labels, num_factors))
    denominators = np.zeros(num_labels)
    ps = np.zeros(num_labels)

    for i in range(n):
        label_index = unique_labels.index(labels[i])

        numerators[label_index] += factors[i]
        denominators[label_index] += sum(factors[i])
        ps[label_index] += 1

    return numerators, denominators, ps / float(n), unique_labels


def classify(new_xs, numerators, denominators, ps, labels):
    n = np.array(numerators) + 1
    d = [[d, d] for d in np.array(denominators) + 2]
    vectors = [sum(row) for row in np.log(n/d) * np.tile(new_xs, (len(labels), 1))] + np.log(ps)

    return vectors, labels


