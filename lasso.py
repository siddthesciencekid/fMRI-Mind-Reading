# Siddhartha Gorti & Shilpa Kumar
# Final Project
# CSE 446 Machine Learning
# WINTER 2017

# Implementation of the standard LASSO algorithm

import numpy as np


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
            # to see if the change was bigger than our epsilon of convergence
            new_weight = soft_threshold_lasso(a_j, c_j, lmbda)
            if abs(weights[j] - new_weight) > 10 ** -6:
                converged = False
            weights[j] = new_weight
    return weights


# Defines the soft threshold for standard LASSO
def soft_threshold_lasso(a_j, c_j, lmbda):
    if c_j < -lmbda:
        return (c_j + lmbda) / a_j
    elif -lmbda <= c_j <= lmbda:
        return 0
    else:
        return (c_j - lmbda) / a_j