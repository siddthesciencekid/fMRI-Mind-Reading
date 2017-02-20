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


   # lambdaValues = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, .4]
   # results = [0, 0, 0, 0, 0, 0, 0, 0]
   # results[0] = squared_error(y, signals_train, scd(.05, y, signals_train, weights, 20))
   # results[1] = squared_error(y, signals_train, scd(.1, y, signals_train, weights, 20))
   # results[2] = squared_error(y, signals_train, scd(.15, y, signals_train, weights, 20))
   # results[3] = squared_error(y, signals_train, scd(.2, y, signals_train, weights, 20))
   # results[4] = squared_error(y, signals_train, scd(.25, y, signals_train, weights, 20))
   # results[5] = squared_error(y, signals_train, scd(.3, y, signals_train, weights, 20))
   # results[6] = squared_error(y, signals_train, scd(.35, y, signals_train, weights, 20))
   # results[7] = squared_error(y, signals_train, scd(.4, y, signals_train, weights, 20))

   # plt.plot(lambdaValues, results)
   # plt.savefig('squaredErrorTraining.png')
   # plt.close()

    print(squared_error(pgd(10, y, signals_train, weights, 1), y, signals_train))
    print(squared_error(pgd(100, y, signals_train, weights, 1), y, signals_train))


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

        print(current_iteration)

    return w


def m_func(A, x_k, b, step_length, x):
    part_1 = f_func(A,x_k,b)
    part_2 = np.dot( (np.dot(A.T, np.dot(A,x_k.T) - b)).T, (x-x_k).T )
    part_3 = (np.linalg.norm(x-x_k)**2) / (2*step_length)
    m_value = part_1 + part_2 + part_3
    return m_value

def f_func(A, x, b):
	f_value = (np.linalg.norm(np.dot(A,x.T) - b)**2) / 2
	return f_value

def sign(y):
	cy = np.copy(y)
	length = len(cy)
	for i in range(length):
		if cy[i] > 0.0:
			cy[i] = 1.0
		elif cy[i] < 0.0:
			cy[i] = -1.0
		else:
			cy[i] = 0.0
	return cy

# Proximal Gradient Descent for LASSO
def pgd(lmbda, y, X, w, step_size):
	current_iteration = 0
	num_iterations = 2000
	w_new = np.copy(w)

	while (current_iteration < num_iterations):
		# Compute gradients while picking correct stepsize
		f_value = 100
		m_value = 0
		while (f_value > m_value):
			gradient = w - step_size * np.dot(X.T, np.dot(X, w) - y)
			w_new = sign(gradient) * vector_max(0, np.absolute(gradient) - step_size * lmbda)
			print(w_new)
			f_value = f_func(X, w_new, y)
			m_value = m_func(X, w, y, step_size, w_new)
			step_size = step_size * .5

		if f_func(X, w, y) <= f_func(X, w_new, y):
			current_iteration = num_iterations
		else:
			w = np.copy(w_new)

	return w

def vector_max(value, vector):
	for i in range(len(vector)):
		vector[i] = max(value, vector[i])
	return vector

def squared_error(weights, y, X):
    total_error = 0
    for i in range(0, len(y)):
        cur_row = X[i, :]
        error = (y[i] - np.dot(cur_row, weights)) ** 2
        total_error += error
    return total_error

if __name__ == "__main__":
    main()
