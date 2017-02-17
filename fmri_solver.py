# Siddhartha Gorti & Shilpa Kumar
# FMRI Project
# CSE 446 Machine Learning
# WINTER 2017

import scipy.io
import numpy as np
from random import randint


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

    # Build the y component of the lasso algorithm
    # for the first semantic feature

    y = np.zeros([len(words_train)])


    for i in range(len(words_train)):
        word_index = words_train[i]
        y[i] = semantic_features[word_index - 1][0]

    y = np.asarray(y)
    print(len(signals_train[0]))
    weights = np.zeros([len(signals_train[0])])
    z = np.zeros([len(y)])

    weights = scd(10, y, signals_train, weights, z, 1, .01)

    print(weights)
    print(np.count_nonzero(weights))


    print(weights)
    print(np.count_nonzero(weights))

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
            # Compute y_i - wTX + the weight at j for the particular entry
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


def scd(lmbda, y, X, w, z, num_epochs, step_size):
    for t in range(num_epochs):
        # Pick a random j
        for j in range(len(X[0])):
            beta = 1 # 1 for squared loss
            sum = 0
            for i in range(len(y)):
                if X[i][j] != 0:
                    sum += (loss_prime(z[i], y[i]) * X[i][j])
            g_j = (1/float(len(y))) * sum
            update_term = w[j] - g_j / beta
            print(update_term)
            if update_term > (lmbda / beta):
                w[j] = update_term - (lmbda / beta)
            elif update_term < -(lmbda / beta):
                w[j] = update_term + (lmbda / beta)
            else:
                w[j] = 0
            for i in range(len(y)):
                if X[i][j] != 0:
                    z[i] = z[i] + step_size * X[i][j]

    return w

# Squared loss
def loss_prime(a, y):
    return a - y

if __name__ == "__main__":
    main()
