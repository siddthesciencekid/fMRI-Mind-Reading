# Siddhartha Gorti & Shilpa Kumar
# FMRI Project
# CSE 446 Machine Learning
# WINTER 2017

import scipy.io
import numpy as np
from random import randint

from matplotlib import pyplot as plt


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
    weights = np.zeros([len(signals_train[0])])
    z = np.zeros([len(y)])

    lambdaValues = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, .4]
    results = [0, 0, 0, 0, 0, 0, 0, 0]
    results[0] = squared_error(y, signals_train, scd(.05, y, signals_train, weights, z, 10))
    results[1] = squared_error(y, signals_train, scd(.1, y, signals_train, weights, z, 10))
    results[2] = squared_error(y, signals_train, scd(.15, y, signals_train, weights, z, 10))
    results[3] = squared_error(y, signals_train, scd(.2, y, signals_train, weights, z, 10))
    results[4] = squared_error(y, signals_train, scd(.25, y, signals_train, weights, z, 10))
    results[5] = squared_error(y, signals_train, scd(.3, y, signals_train, weights, z, 10))
    results[6] = squared_error(y, signals_train, scd(.35, y, signals_train, weights, z, 10))
    results[7] = squared_error(y, signals_train, scd(.4, y, signals_train, weights, z, 10))

    plt.plot(lambdaValues, results)
    plt.savefig('squaredErrorTraining.png')
    plt.close()


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

def scd(lmbda, y, X, w, z, num_iterations):
    num_attributes = len(X[0])
    num_values = len(y)
    current_iteration = 0

    w_minus = np.zeros([num_attributes]) # 21xxx x 1 vector
    w_plus = np.zeros([num_attributes]) # 21xxx x 1 vector
    w_old = np.copy(w) # 21xxx x 1 vector
    wTX = np.dot(X, w) # 300 x 1 vector

    # While not converged
    while current_iteration < num_iterations:
        # Each iteration, do num_attributes worth updates
        for i in range(num_attributes):
            # Pick a random j from 0 to 2*num_attributes - 1
            j = randint(0, (num_attributes * 2) - 1)

            if (j < num_attributes):
                x_j = X[:, j] # 300 x 1 vector
                diff = np.subtract(wTX, y)
                gradient = np.dot(x_j, diff) / num_values * 1.0
                max_add = max(-w_plus[j], -gradient - lmbda)
                w_plus[j] += max_add
                wTX_change = x_j * max_add
                wTX = np.add(wTX, wTX_change)
            else:
                j = j - num_attributes
                x_j = X[:, j]
                diff = np.subtract(wTX, y)
                gradient = -1.0 * np.dot(x_j, diff) / num_values
                max_add = max(-w_minus[j], -gradient - lmbda)
                w_minus[j] +=  max_add
                wTX_change = x_j * -max_add
                wTX = np.add(wTX, wTX_change)

        current_iteration += 1
        w = np.subtract(w_plus, w_minus)

        if np.linalg.norm(np.subtract(w, w_old)) < 10 ** -6:
            break
        w_old = np.copy(w)

        print(current_iteration)

    return w




# Squared loss
def loss_prime(a, y):
    return a - y

def squared_error(y, X, w):
    y2 = np.copy(y)
    for i in range(X.shape[0]):
        y2[i] = np.dot(X[i], w)

    diff = (y-y2)
    sumError = np.sum(np.square(diff))
    return (sumError / X.shape[0])

if __name__ == "__main__":
    main()
