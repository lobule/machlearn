import data, knn, trees


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

#data, labels = data.create_bird_set()
#print(labels)

#print(data)
#print(trees.calculate_entropy(data))
#print(trees.calculate_entropy([0, 1]))

#print(trees.split(data, 0, 1))
#print(trees.split(data, 0, 0))

#print(trees.find_optimal_split_feature(data))

#print(trees.get_majority([1, 1, 2, 3, 3, 3]))

#tree = trees.create_tree(data, labels)

#trees.save_tree(tree, "savetree.txt")
#print(trees.load_tree("savetree.txt"))

matrix, labels = data.create_contacts_set()
lens_tree = trees.create_tree(matrix, labels)
print(lens_tree)

#print(trees.classify([1, 0], tree, labels))
#print(trees.classify([1, 1], tree, labels))
