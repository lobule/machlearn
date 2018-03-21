from unittest import TestCase
import knn
import trees
import data
import numpy as np


class TestClassify(TestCase):
    def test_knn_classify_basic(self):
        factors = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1], [-1, -1], [-0.5, -1.1]])
        labels = ['A', 'A', 'B', 'B', 'C', 'C']

        prediction = knn.classify([1.1, 0.9], factors, labels, 3)
        self.failUnless(prediction == 'A')

        prediction = knn.classify([-0.1, 0.3], factors, labels, 3)
        self.failUnless(prediction == 'B')

        prediction = knn.classify([-0.7, -1.3], factors, labels, 3)
        self.failUnless(prediction == 'C')

    def test_knn_classify_dating(self):
        xs, ys = data.create_dating_set()
        xs, mins, ranges = knn.normalize_cols(xs)

        n = xs.shape[0]
        test_n = int(n * 0.1)

        errors = 0
        for i in range(test_n):
            predicted_y = knn.classify(xs[i], xs[test_n:n, :], ys[test_n:n], 5)
            if predicted_y != ys[i]:
                errors += 1

        print(errors)
        print(test_n)
        self.failUnless(errors/float(test_n) < 0.85)

    def test_knn_classify_digits(self):
        test_xs, test_ys = data.create_digit_matrix('digits/testDigits')
        train_xs, train_ys = data.create_digit_matrix('digits/trainingDigits')

        errors = 0
        for index, test_row in enumerate(test_xs):
            predicted_y = knn.classify(test_row, train_xs, train_ys, 3)
            if predicted_y != test_ys[index]:
                errors += 1

        print(errors)
        print(test_xs.shape[0])
        self.failUnless(errors/test_xs.shape[0] < 0.85)

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

    def test_trees_classify_lenses(self):
        matrix, labels = data.create_contacts_set()

        lens_tree = trees.create_tree(matrix, labels)

        filename = "save_lens_tree.txt"
        trees.save_tree(lens_tree, filename)

        self.failUnless(True)
