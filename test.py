import unittest
import numpy as np
import trees
import math
import knn
import bayes


class Test(unittest.TestCase):

    def test_calculate_entropy(self):
        self.failUnless(trees.calculate_entropy([0]) == 0)
        self.failUnless(trees.calculate_entropy([0, 0, 0, 1, 1, 1]) == -math.log(0.5, 2))
        self.failUnless(trees.calculate_entropy(['A', 'B']) == -math.log(0.5, 2))
        self.failUnless(trees.calculate_entropy(['A', 'B', 'C', 'D', 'E']) == -math.log(0.2, 2))
        self.failUnless(trees.calculate_entropy([0, 0, 'A', 'A', 'A']) ==
                        -0.4 * math.log(0.4, 2) +
                        -0.6 * math.log(0.6, 2))



    def test_get_token_set(self):
        corpus = [['this', 'is', 'not', 'a', 'test'],
                  ['actually', 'this', 'is', 'a', 'test']]

        vocabulary = bayes.get_token_set(corpus)
        self.failUnless(set(vocabulary) == {'this', 'is', 'not', 'actually', 'a', 'test'})
        self.failUnless(len(vocabulary) == 6)

    def test_get_vocabulary_set_vector(self):
        vocabulary = 'these are the words'.split(' ')

        sample = 'these are words'.split(' ')
        self.failUnless(bayes.get_vocabulary_set_vector(vocabulary, sample) == [1, 1, 0, 1])

        sample = 'these words may really be words'.split(' ')
        self.failUnless(bayes.get_vocabulary_set_vector(vocabulary, sample) == [1, 0, 0, 1])

        vocabulary = []
        sample = 'edge case test'.split(' ')
        self.failUnless(bayes.get_vocabulary_set_vector(vocabulary, sample) == [])

    def test_get_vocabulary_bag_vector(self):
        vocabulary = 'these are the words'.split(' ')

        sample = 'these are words'.split(' ')
        self.failUnless(np.all(bayes.get_vocabulary_bag_vector(vocabulary, sample) == [1, 1, 0, 1]))

        sample = 'these words are words but not the words'.split(' ')
        self.failUnless(np.all(bayes.get_vocabulary_bag_vector(vocabulary, sample) == [1, 1, 1, 3]))

        vocabulary = []
        sample = 'edge case test'.split(' ')
        self.failUnless(np.all(bayes.get_vocabulary_bag_vector(vocabulary, sample) == []))

    def test_parse_to_word_list(self):
        self.failUnless(bayes.parse_to_word_list('') == [])
        self.failUnless(bayes.parse_to_word_list('SiNgLeToN') == ['singleton'])

        text = 'Some words, including punctuation, AND capitalization, and extra    spaces.'
        words = ['some', 'words', 'including', 'punctuation', 'and', 'capitalization', 'and', 'extra', 'spaces']
        self.failUnless(bayes.parse_to_word_list(text) == words)

    def test_get_max_valued_key(self):
        tally = {'two': 2, 'six': 6, 'one': 1, 'four': 4}
        self.failUnless(knn.get_most_common(tally) == 'six')

    def test_train_naive_bayes(self):
        factors = np.array([[1, 0], [1, 0], [0, 1], [1, 0], [0, 1]])
        labels = ['A', 'A', 'B', 'B', 'C', ]

        numerators, denominators, ps, unique_labels = bayes.train_naive_bayes(factors, labels)

        self.failUnless(set(unique_labels) == set(['A', 'B', 'C']))
        self.failUnless(np.all(numerators[unique_labels.index('A')] == [2, 0]))
        self.failUnless(np.all(numerators[unique_labels.index('B')] == [1, 1]))
        self.failUnless(np.all(numerators[unique_labels.index('C')] == [0, 1]))

        self.failUnless(np.all(denominators[unique_labels.index('A')] == 2))
        self.failUnless(np.all(denominators[unique_labels.index('B')] == 2))
        self.failUnless(np.all(denominators[unique_labels.index('C')] == 1))

        self.failUnless(ps[unique_labels.index('A')] == 0.4)
        self.failUnless(ps[unique_labels.index('B')] == 0.4)
        self.failUnless(ps[unique_labels.index('C')] == 0.2)



