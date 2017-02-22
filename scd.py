# Siddhartha Gorti & Shilpa Kumar
# Final Project
# CSE 446 Machine Learning
# WINTER 2017

# Implementation of the Stochastic Coordinate Descent for LASSO

import numpy as np
from random import randint

# Stochastic Coordinate Descent for Lasso
def scd(lmbda, y, X, w, num_iterations):
    num_attributes = len(X[0])
    num_values = len(y)
    current_iteration = 0
    # To increase chances of convergence , make each iteration have num_attributes # of updates
    num_iterations = num_iterations * num_attributes
    w_tilda = np.zeros([num_attributes * 2]) # w~ = w+ -- w-
    negX = X * -1
    X_tilda = np.hstack((X, negX))
    # Keep track of wTX so that you only compute dot product once
    wTX = np.dot(X, w) # 300 x 1 vector

    # While not converged
    while current_iteration < num_iterations:
        # Pick a random j from 0 to 2*num_attributes - 1
        j = randint(0, (num_attributes * 2) - 1)

        # ** We could use w = w_plus - w_minus and compute the dot product for the gradient every time, since
        # we update w_plus and w_minus after every random pick of j. However, computing the dot product every single
        # run of every iteration takes very long. So we will keep track of w_transpose * X, and when we update w_plus and w_minus,
        # we either add w_plus * x_j or subtract w_minus * x_j >> leads to the same thing much faster. **

        diff = np.subtract(wTX, y)
        # Get jth column
        x_j = X_tilda[:, j]

        # Gradient = 1/n * x_j * (difference) + lambda
        gradient = 1.0 * np.dot(x_j, diff) / num_values + lmbda
        # Max of -w~[j] and -gradient
        max_add = max(-w_tilda[j], -gradient)
        # Update w~[j] (in this case w_minus[j])
        w_tilda[j] +=  max_add

        # Update wTX instead of redoing dot product to save time
        wTX = np.add(wTX, max_add * x_j)

        # Update iteration
        current_iteration += 1

    # w~ = w_plus - w_minus
    # We don't keep track of this earlier because it is not used for dot products bc of lack of efficiency.
    split = np.split(w_tilda, 2)
    w = np.subtract(split[0], split[1])
    return w