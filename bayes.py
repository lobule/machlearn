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
    num_factors = len(factors[0])

    unique_labels = list(set(labels))
    num_labels = len(unique_labels)

    numerator = tile(0, (num_labels, num_factors))

    denominator = zeros(num_labels)

    for i in range(len(factors)):
        label_index = unique_labels.index(labels[i])

        numerator[label_index] += factors[i]
        denominator[label_index] += sum(factors[i])

    return numerator, denominator, unique_labels



