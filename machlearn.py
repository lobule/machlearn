import data
import knn
import trees

def knn_dating_test():
    xs, ys = data.create_dating_set()
    xs, mins, ranges = data.normalize_cols(xs)

    n = xs.shape[0]
    test_n = int(n * 0.1)

    errors = 0
    for i in range(test_n):
        predicted_y = knn.classify(xs[i], xs[test_n:n, :], ys[test_n:n], 5)
        if predicted_y != ys[i]:
            errors += 1

    print(errors)
    print(test_n)
    print(errors/float(test_n))


def knn_digit_test():

    test_xs, test_ys = data.create_digit_matrix('digits/testDigits')
    train_xs, train_ys = data.create_digit_matrix('digits/trainingDigits')

    errors = 0
    for index, test_row in enumerate(test_xs):
        predicted_y = knn.classify(test_row, train_xs, train_ys, 3)
        if predicted_y != test_ys[index]:
            errors += 1

    print(errors)
    print(test_xs.shape[0])
    print(errors/test_xs.shape[0])


def trees_lens_test():
    matrix, labels = data.create_contacts_set()

    lens_tree = trees.create_tree(matrix, labels)

    filename = "save_lens_tree.txt"
    trees.save_tree(lens_tree, filename)

    print(trees.load_tree(filename))