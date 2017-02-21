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
    # our total cost is f(x) + lambda * l1 penalty = RSS(w) + lambda*l1penalty
    # References to f(x) are to the RSS(w)
    while (current_iteration < num_iterations):
        # Compute gradients while picking correct stepsize
        f_xk1 = 100
        line_search_value = 0
        # Pick correct stepsize using Armijo's rule, ends when f(x^k+1) <= f(x^k) - 1/2(1/step_size)*||gradient(f(x^k))||^2
        while (f_xk1 > line_search_value):
            # Compute the gradient vector. X^T*(Xw - y)
            # In the book this is described as g_k
            g_k = np.dot(X.T, np.dot(X, w) - y)
            # Subtract gradient * step_size from current vector of w
            # In the book this is described as u_k
            u_k = w - step_size * g_k

            # Soft thresholding element-wise: soft(gradient, step_size * lambda)
            # Since the penalty is L1, this is the proximal operator.
            w_new = np.sign(u_k) * v_max(np.absolute(u_k) - step_size * lmbda)

            # Use line search to update the step size
            # Using Armijo's rule
            # Stop when f(x^k+1) <= f(x^k) - 1/2(1/step_size)*||gradient(f(x^k))||^2
            f_xk1 = res_sum_square(X, w_new, y)
            f_xk = res_sum_square(X, w, y)
            line_search_value = line_search(X, w, y, step_size, w_new)
            step_size = step_size * beta

        # Converged, exit while loop
        if res_sum_square(X, w, y) <= res_sum_square(X, w_new, y):
            current_iteration = num_iterations
        else:
            w = np.copy(w_new)
    return w


def line_search(X, w, y, step_size, w_new):
    f_xk = res_sum_square(X, w, y)
    fit_gradient = np.dot((np.dot(X.T, np.dot(X,w.T) - y)).T, (w_new - w).T)
    fit_fx = (np.linalg.norm(w_new - w) ** 2) / (2 * step_size)
    return f_xk + fit_gradient + fit_fx


# RSS: ||wX - y||^2 (Measure of fit)
def res_sum_square(X, w, y):
    return (np.linalg.norm(np.dot(X, w.T) - y)**2)


# Runs through the vector and changes all negative values
# to zero
def v_max(vector):
    for i in range(len(vector)):
        vector[i] = max(0, vector[i])
    return vector