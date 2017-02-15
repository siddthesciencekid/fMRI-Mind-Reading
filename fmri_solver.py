# Siddhartha Gorti & Shilpa Kumar
# FMRI Project
# CSE 446 Machine Learning
# WINTER 2017

import scipy.io
import numpy as np


def import_words_train(file_name, array):
    f = open(file_name)
    index = 0
    for line in f:
        temp = line
        array[index] = int((temp.split())[0])
        index += 1
    return array


def main():
    signals_test = scipy.io.mmread("data/subject1_fmri_std.test.mtx")
    signals_train = scipy.io.mmread("data/subject1_fmri_std.train.mtx")
    words_test = scipy.io.mmread("data/subject1_wordid.test.mtx")

    # Training wordid only has 1 column, and using mmread doesn't work
    words_train = [0 for x in range(300)]
    words_train = import_words_train("data/subject1_wordid.train.mtx", words_train)
    words_train = np.asarray(words_train)
    
    semantic_features = scipy.io.mmread("data/word_feature_centered.mtx")

    signals_train = np.asarray(signals_train)

    # Build the y component of the lasso algorithm
    # for the first semantic feature

    y = np.zeros([len(words_train), 1])


    for i in range(0, len(words_train)):
        word_index = words_train[i]
        y[i] = semantic_features[word_index - 1][1]

    y = np.asarray(y)

    weights = np.zeros([len(signals_train[0]), 1])

    weights = lasso(0.8, y, signals_train, weights)

    print(weights)
    print(signals_train.shape)
    print(signals_test.shape)
    print(words_train.shape)
    print(words_test.shape)
    print(semantic_features.shape)


def soft_threshold(a_j, c_j, lmbda):
    if c_j < -lmbda:
        return (c_j + lmbda) / a_j
    elif -lmbda <= c_j <= lmbda:
        return 0
    else:
        return (c_j - lmbda) / a_j


def lasso(lmbda, y, X, weights):
    converged = False
    while not converged:
        converged = True
        for j in range(0, len(X[0])):
            cur_column = X[:, j]
            # Compute the a term for the current column
            a_j = 2 * np.sum(cur_column ** 2)
            # Compute the W Transpose times all rows in X
            wTX = np.dot(X, weights)
            # Compute y_i - wTX + the wieght at j for the particular entry
            temp = y - wTX + np.multiply(weights[j], cur_column)
            # Finally compute c_j
            c_j = 2 * np.sum(cur_column * temp)

            # Get the new weight for this entry and check
            # to see if the change was bigger than our epsilon
            new_weight = soft_threshold(a_j, c_j, lmbda)
            if abs(weights[j] - new_weight) > 10 ** -6:
                converged = False
            weights[j] = new_weight
    return weights

if __name__ == "__main__":
    main()
