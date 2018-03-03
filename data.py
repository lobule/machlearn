from numpy import *
from pandas import Categorical
from os import listdir

import numpy as np

#import matplotlib
#import matplotlib.pyplot as plt
#matplotlib.use('TkAgg')

def create_data_set():
    factors = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return factors, labels


def create_dating_set():
    reader = open("datingTestSet.txt")
    content = reader.readlines()
    factors = []
    labels = []
    for line in content:
        line = line.strip().split('\t')
        factors.append([float(f) for f in line[0:-1]])
        labels.append(line[-1])
    factors = np.array(factors)

    return factors, labels

def create_contacts_set():
    reader = open("lenses.txt")
    matrix = [row.strip().split('\t') for row in reader.readlines()]
    labels = ['age', 'prescription', 'astigmatic', 'tear_rate']
    return matrix, labels

def create_digit_vector(filename):
    reader = open(filename)
    content = reader.readlines()
    vector = []
    for line in content:
        vector += [int(i) for i in line[0:-1]]
    return vector


def create_digit_matrix(folder_name):
    file_names = listdir(folder_name)
    n = len(file_names)
    digit_matrix = []
    labels_vector = []
    for i in range(n):
        file_name = file_names[i]
        class_name = int(file_name.split('_')[0])
        data = create_digit_vector(folder_name + '/' + file_name)
        digit_matrix.append(data)
        labels_vector += [class_name]
    digit_matrix = np.array(digit_matrix)

    return digit_matrix, labels_vector


def normalize_cols(matrix):
    mins = matrix.min(0)
    maxes = matrix.max(0)
    ranges = maxes - mins

    rows = matrix.shape[0]
    normal_matrix = (matrix - tile(mins, (rows, 1))) / tile(ranges, (rows, 1))

    return normal_matrix, mins, ranges


def plot(factors, labels):
    rows = factors[:, 0]
    cols = factors[:, 1]
    codes = Categorical(labels).codes

    #figure = plt.figure().add_subplot(111)
    #figure.scatter(rows, cols, codes + 1, ['red' if c == 1 else 'green' if c == 2 else 'blue' for c in codes])
    #plt.show()


def create_bird_set():
    matrix = [[1, 1, 'yes'],
              [1, 1, 'yes'],
              [1, 0, 'no'],
              [0, 1, 'no'],
              [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return matrix, labels









