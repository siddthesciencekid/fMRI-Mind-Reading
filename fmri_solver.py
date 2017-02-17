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

    lambdaValues = [30, 15, 10, 1, .5]
    results = [0, 0, 0, 0, 0]
    print("30")
    results[0] = squared_error(y, signals_train, scd_2(30, y, signals_train, weights, z, 40000, .01))
    print(results[0])

    print("15")
    results[1] = squared_error(y, signals_train, scd_2(15, y, signals_train, weights, z, 40000, .01))
    print(results[1])

    print("10")
    results[2] = squared_error(y, signals_train, scd_2(10, y, signals_train, weights, z, 40000, .01))
    print(results[2])

    print("1")
    results[3] = squared_error(y, signals_train, scd_2(1, y, signals_train, weights, z, 40000, .01))

    print(".5")
    results[4] = squared_error(y, signals_train, scd_2(.5, y, signals_train, weights, z, 40000, .01))

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


def scd(lmbda, y, X, w, z, num_epochs, step_size):
    for t in range(num_epochs):
        # Pick a random j
        j = randint(0, len(X[0]) - 1)
        beta = 1 # 1 for squared loss
        sum = 0
        for i in range(len(y)):
            if X[i][j] != 0:
                sum += (loss_prime(z[i], y[i]) * X[i][j])
        g_j = (1/float(len(y))) * sum
        update_term = w[j] - g_j / beta
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


def scd_2(lmbda, y, X, w, z, num_iterations, step_size):
    num_attributes = len(X[0])
    w_minus = np.zeros([num_attributes])
    w_plus = np.zeros([num_attributes])
    wTX = np.dot(X, w)
    num_values = len(y)
    # While not converged
    for i in range(num_iterations):
        # Choose j at random
        j = randint(0, (num_attributes * 2) - 1)
        #-1/n(Y - X~w~) * x_j  + lambda

        # J can be anywhere from 0 to 2p-1, so check which half it's on, and subsequently update correct 'side' of w. 
        if (j < num_attributes):
            x_j = X[:, j]
            diff = np.subtract(y, wTX)
            gradient = np.dot(x_j, diff) / num_values * 1.0
            w_plus[j] += max(-w_plus[j], -gradient - lmbda)
            wTX_change = x_j * (max(-w_plus[j], -gradient - lmbda))
            wTX = np.add(wTX, wTX_change)
        else:
            j = j % num_attributes
            x_j = X[:, j]
            diff = np.subtract(y, wTX)
            gradient = -np.dot(x_j, diff) / num_values * 1.0
            w_minus[j] +=  max(-w_minus[j], -gradient - lmbda)
            wTX_change = x_j * (-max(-w_minus[j], -gradient - lmbda))
            wTX = np.add(wTX, wTX_change)

    return np.subtract(w_plus, w_minus)

# Squared loss
def loss_prime(a, y):
    return a - y

def squared_error(y, X, w):
    y2 = np.copy(y)
    for i in range(X.shape[0]):
        y2[i] = np.dot(X[i], w)

    diff = (y-y2)
    print(y)
    print(y2)
    sumError = np.sum(np.square(diff))
    return (sumError / X.shape[0])

if __name__ == "__main__":
    main()
