# Siddhartha Gorti & Shilpa Kumar
# FMRI Project
# CSE 446 Machine Learning
# WINTER 2017

import scipy.io
import numpy as np
from random import randint
import timeit
from matplotlib import pyplot as plt_training
from matplotlib import pyplot as plt_PGD
from matplotlib import pyplot as plt_SCD

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


def soft_threshold_lasso(a_j, c_j, lmbda):
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
            new_weight = soft_threshold_lasso(a_j, c_j, lmbda)
            if abs(weights[j] - new_weight) > 10 ** -6:
                converged = False
            weights[j] = new_weight
    return weights

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

def line_search(X, w, y, step_size, w_new):
    f_xk = res_sum_square(X, w, y)
    fit_gradient = np.dot((np.dot(X.T, np.dot(X,w.T) - y)).T, (w_new - w).T)
    fit_fx = (np.linalg.norm(w_new - w) ** 2) / (2 * step_size)
    return f_xk + fit_gradient + fit_fx

# RSS: ||wX - y||^2 (Measure of fit)
def res_sum_square(X, w, y):
	return (np.linalg.norm(np.dot(X, w.T) - y)**2)

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

def v_max(vector):
	for i in range(len(vector)):
		vector[i] = max(0, vector[i])
	return vector

def squared_error(y, X, weights):
	y2 = np.copy(y)
	for i in range(X.shape[0]):
		y2[i] = np.dot(X[i], weights)

	diff = y-y2
	sumError = np.sum(np.square(diff))
	return sumError / X.shape[0]


if __name__ == "__main__":
    main()
