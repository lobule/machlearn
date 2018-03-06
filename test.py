import unittest
import numpy
import knn


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
