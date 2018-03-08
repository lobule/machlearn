def get_token_set(corpus):
    tokens = set([])
    for sample in corpus:
        tokens = tokens | set(sample)
    return tokens

