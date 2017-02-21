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

    w_minus = np.zeros([num_attributes]) # 21xxx x 1 vector
    w_plus = np.zeros([num_attributes]) # 21xxx x 1 vector
    w_old = np.copy(w) # 21xxx x 1 vector
    # Keep track of wTX so that you only compute dot product once
    wTX = np.dot(X, w) # 300 x 1 vector

    # While not converged
    while current_iteration < num_iterations:
        # Each iteration, do num_attributes worth updates
        for i in range(num_attributes):
            # Pick a random j from 0 to 2*num_attributes - 1
            j = randint(0, (num_attributes * 2) - 1)

            # ** Since w~ is [w_plus; w_minus], and X~ is [X, -X], we save a lot of space by doing the algorithm in terms of just X without replicating
            # it. So, to account for this, we split w~ into w_plus and w_minus.
            # We have two cases: where j is in the 'left half', and where it is in the 'right half'. When it is in the left half,
            # j < num_attributes, and we update w_plus. In the other case, we update w_minus, and because it is -X, the gradient's sign is flipped. **

            # ** We could use w = w_plus - w_minus and compute the dot product for the gradient every time, since
            # we update w_plus and w_minus after every random pick of j. However, computing the dot product every single
            # run of every iteration takes very long. So we will keep track of w_transpose * X, and when we update w_plus and w_minus,
            # we either add w_plus * x_j or subtract w_minus * x_j >> leads to the same thing much faster. **

            # Left half so w(plus)
            if (j < num_attributes):
                # Get jth column
                x_j = X[:, j]
                # Gradient = 1/n * x_j * (xw~ - y) + lambda
                diff = np.subtract(wTX, y)
                gradient = np.dot(x_j, diff) / num_values * 1.0 + lmbda
                # Max of -w~[j] and -gradient. Since j < num_attributes, update w_plus
                max_add = max(-w_plus[j], -gradient)
                # Update w~[j] (in this case w_plus[j])
                w_plus[j] += max_add
                # Only one column of w changed so update x_j column with new weight and add to wTX
                wTX = np.add(wTX, max_add * x_j)

            # Right half so w(minus)
            else:
                # Align j properly
                j = j - num_attributes
                # Get jth column
                x_j = X[:, j]
                # Gradient = 1/n * x_j * (y - w~X) + lambda
                diff = np.subtract(y, wTX)
                gradient = 1.0 * np.dot(x_j, diff) / num_values + lmbda
                # Max of -w~[j] and -gradient. Since j >= p, update w_minus
                max_add = max(-w_minus[j], -gradient)
                # Update w~[j] (in this case w_minus[j])
                w_minus[j] +=  max_add
                # W = w_plus - w_minus. So to update completely, subtract correct amount
                # Only one column of w changed so update x_j column with new weight and subtract from wTX
                wTX = np.subtract(wTX, max_add * x_j)

        current_iteration += 1
        # w~ = w_plus - w_minus
        # We don't keep track of this earlier because it is not used for dot products bc of lack of efficiency.
        w = np.subtract(w_plus, w_minus)

        # Hit convergence before all the iterations are over
        if np.linalg.norm(np.subtract(w, w_old)) < 10 ** -6:
            break
        w_old = np.copy(w)

    return w