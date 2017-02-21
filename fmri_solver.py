# Siddhartha Gorti & Shilpa Kumar
# Final Project
# CSE 446 Machine Learning
# WINTER 2017

import scipy.io
import numpy as np
import timeit

from matplotlib import pyplot as plt_training
from matplotlib import pyplot as plt_PGD
from matplotlib import pyplot as plt_SCD
from lasso import lasso
from scd import scd
from pgd import pgd


def main():
    signals_test = scipy.io.mmread("data/subject1_fmri_std.test.mtx")
    signals_train = scipy.io.mmread("data/subject1_fmri_std.train.mtx")
    words_test = scipy.io.mmread("data/subject1_wordid.test.mtx")

    # Training wordid only has 1 column, and using mmread doesn't work
    words_train = [0 for x in range(300)]
    words_train = import_words_train("data/subject1_wordid.train.mtx", words_train)
    words_train = np.asarray(words_train)
    
    semantic_features = scipy.io.mmread("data/word_feature_centered.mtx")

    # Build the y component of the lasso algorithm for the 200th semantic feature
    y = np.zeros([len(words_train)])
    for i in range(len(words_train)):
        word_index = words_train[i]
        y[i] = semantic_features[word_index - 1][199]

    # Build the y vector for test data
    y_test = np.zeros([len(words_test)])
    for i in range(len(words_test)):
        word_index = words_test[i][0]
        # Word test stores things as floats, and the lookup in semantic features doesn't work
        # unless it is converted to int
        word_index = int(word_index)
        y_test[i] = semantic_features[word_index - 1][199]
 

    y = np.asarray(y)
    y_test = np.asarray(y_test)

    weights = np.zeros([len(signals_train[0])])

    lambdaValues = [.1, .5, 1, 5, 10, 20, 40, 100, 200]
    results = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    results[0] = squared_error(y, signals_train, pgd(.1, y, signals_train, weights, 20, 20))
    results[1] = squared_error(y, signals_train, pgd(.5, y, signals_train, weights, 20, 20))
    results[2] = squared_error(y, signals_train, pgd(1, y, signals_train, weights, 20, 20))
    results[3] = squared_error(y, signals_train, pgd(5, y, signals_train, weights, 20, 20))
    results[4] = squared_error(y, signals_train, pgd(10, y, signals_train, weights, 20, 20))
    results[5] = squared_error(y, signals_train, pgd(20, y, signals_train, weights, 20, 20))
    results[6] = squared_error(y, signals_train, pgd(40, y, signals_train, weights, 20, 20))
    results[7] = squared_error(y, signals_train, pgd(100, y, signals_train, weights, 20, 20))
    results[8] = squared_error(y, signals_train, pgd(200, y, signals_train, weights, 20, 20))

    plt_PGD.plot(lambdaValues, results, label = "PGD")
    plt_PGD.savefig('squaredErrorTrain_PGD_sem200.png')
    plt_PGD.close()

    results = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    results[0] = squared_error(y_test, signals_test, pgd(.1, y, signals_train, weights, 20, 20))
    results[1] = squared_error(y_test, signals_test, pgd(.5, y, signals_train, weights, 20, 20))
    results[2] = squared_error(y_test, signals_test, pgd(1, y, signals_train, weights, 20, 20))
    results[3] = squared_error(y_test, signals_test, pgd(5, y, signals_train, weights, 20, 20))
    results[4] = squared_error(y_test, signals_test, pgd(10, y, signals_train, weights, 20, 20))
    results[5] = squared_error(y_test, signals_test, pgd(20, y, signals_train, weights, 20, 20))
    results[6] = squared_error(y_test, signals_test, pgd(40, y, signals_train, weights, 20, 20))
    results[7] = squared_error(y_test, signals_test, pgd(100, y, signals_train, weights, 20, 20))
    results[8] = squared_error(y_test, signals_test, pgd(200, y, signals_train, weights, 20, 20))
    print(results)

    plt_PGD.plot(lambdaValues, results, label = "PGD")
    plt_PGD.savefig('squaredErrorTest_PGD_sem200.png')
    plt_PGD.close()


def import_words_train(file_name, array):
    f = open(file_name)
    index = 0
    for line in f:
        temp = line
        array[index] = int((temp.split())[0])
        index += 1
    return array


def squared_error(y, X, weights):
    y2 = np.copy(y)
    for i in range(X.shape[0]):
        y2[i] = np.dot(X[i], weights)

    diff = y-y2
    sumError = np.sum(np.square(diff))
    return sumError / X.shape[0]


if __name__ == "__main__":
    main()
