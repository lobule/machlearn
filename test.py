import unittest
import numpy
import knn
import trees
import math
import bayes


class TestClassify(unittest.TestCase):
    def test_knn_classify(self):
        factors = numpy.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1], [-1, -1], [-0.5, -1.1]])
        labels = ['A', 'A', 'B', 'B', 'C', 'C']

        prediction = knn.classify([1.1, 0.9], factors, labels, 3)
        self.failUnless(prediction == 'A')

        prediction = knn.classify([-0.1, 0.3], factors, labels, 3)
        self.failUnless(prediction == 'B')

        prediction = knn.classify([-0.7, -1.3], factors, labels, 3)
        self.failUnless(prediction == 'C')

    def test_calculate_entropy(self):
        self.failUnless(trees.calculate_entropy([0]) == 0)
        self.failUnless(trees.calculate_entropy([0, 0, 0, 1, 1, 1]) == -math.log(0.5, 2))
        self.failUnless(trees.calculate_entropy(['A', 'B']) == -math.log(0.5, 2))
        self.failUnless(trees.calculate_entropy(['A', 'B', 'C', 'D', 'E']) == -math.log(0.2, 2))
        self.failUnless(trees.calculate_entropy([0, 0, 'A', 'A', 'A']) ==
                        -0.4 * math.log(0.4, 2) +
                        -0.6 * math.log(0.6, 2))

    def test_trees_classify(self):
        matrix = [[1, 1, 'yes'],
                  [1, 1, 'yes'],
                  [1, 0, 'no'],
                  [0, 1, 'no'],
                  [0, 1, 'no']]
        labels = ['no surfacing', 'flippers']

        tree = trees.create_tree(matrix, labels)

        prediction = trees.classify([1, 1], tree, labels)
        self.failUnless(prediction == 'yes')

        prediction = trees.classify([1, 0], tree, labels)
        self.failUnless(prediction == 'no')

        prediction = trees.classify([0, 1], tree, labels)
        self.failUnless(prediction == 'no')

        prediction = trees.classify([0, 0], tree, labels)
        self.failUnless(prediction == 'no')

    def test_get_token_set(self):
        corpus = [['this', 'is', 'not', 'a', 'test'],
                  ['actually', 'this', 'is', 'a', 'test']]

        vocabulary = bayes.get_token_set(corpus)
        self.failUnless(set(vocabulary) == set(['this', 'is', 'not', 'actually', 'a', 'test']))
        self.failUnless(len(vocabulary) == 6)

    def test_get_vocabulary_vector(self):
        vocabulary = 'these are the words'.split(' ')

        sample = 'these are words'.split(' ')
        self.failUnless(bayes.get_vocabulary_vector(vocabulary, sample) == [1, 1, 0, 1])

        sample = 'these words may really be words'.split(' ')
        self.failUnless(bayes.get_vocabulary_vector(vocabulary, sample) == [1, 0, 0, 1])

        vocabulary = []
        sample = 'edge case test'.split(' ')
        self.failUnless(bayes.get_vocabulary_vector(vocabulary, sample) == [])



