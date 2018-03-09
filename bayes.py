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


