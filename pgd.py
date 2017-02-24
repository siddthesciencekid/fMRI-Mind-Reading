# Siddhartha Gorti & Shilpa Kumar
# Final Project
# CSE 446 Machine Learning
# WINTER 2017

# Implementation of the Proximal Gradient Descent for LASSO

import numpy as np

# Proximal Gradient Descent for LASSO
def pgd(lmbda, y, X, w, step_size, num_iterations):
    current_iteration = 0
    w_new = np.copy(w)
    beta = .5
    alpha = .25
    left = 100
    right = 0
    while (current_iteration < num_iterations):
        while (left > right):
            step_size  = step_size * beta
            g_k = 2.0 * np.dot(X.T, np.dot(X, w) - y)
            u_k = w - step_size * g_k
            w_new = np.sign(u_k) * v_max(np.absolute(u_k) - step_size * lmbda)
            left = res_sum_square(X, w_new, y)
            right = res_sum_square(X, w, y)
            sub = alpha * step_size * np.linalg.norm(g_k) ** 2
            right = right - sub
        w = np.copy(w_new)
        current_iteration += 1
    return w

# RSS: ||wX - y||^2 (Measure of fit)
def res_sum_square(X, w, y):
    return (np.linalg.norm(np.dot(X, w.T) - y)**2)


# Runs through the vector and changes all negative values
# to zero
def v_max(vector):
    for i in range(len(vector)):
        vector[i] = max(0, vector[i])
    return vector