# Siddhartha Gorti & Shilpa Kumar
# Final Project
# CSE 446 Machine Learning
# WINTER 2017

# To build the model with k-fold cross validation call the script with the optional parameter kfold
#
# USAGE INSTRUCTIONS:
# fmri_solver.py <function> <optional_param>
# fmri_solver.py 

import scipy.io
import numpy as np
import sys
import timeit
import math

from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from lasso import lasso
from scd import scd
from pgd import pgd


def main():
    # To use kfold cross validation pass in kfold as a command line parameter
    # Otherwise the best lambda tuning parameter value will be selected using the test set
    # WARNING: ENABLING KFOLD CROSS VALIDATION SIGNIFICANTLY INCREASES THE RUN TIME OF THIS PROGRAM
    if len(sys.argv) > 1:
        use_k_fold = True if sys.argv[1] == "kfold" else False
    else:
        use_k_fold = False

    # READ IN DATA FROM DATA FILES INTO MATRICES
    signals_test = scipy.io.mmread("data/subject1_fmri_std.test.mtx")
    signals_train = scipy.io.mmread("data/subject1_fmri_std.train.mtx")
    words_test = scipy.io.mmread("data/subject1_wordid.test.mtx")

    # Training wordid only has 1 column, and using mmread doesn't work
    words_train = [0 for x in range(300)]
    words_train = import_words_train("data/subject1_wordid.train.mtx", words_train)
    words_train = np.asarray(words_train)
    
    semantic_features = scipy.io.mmread("data/word_feature_centered.mtx")

    # Use these functions to plot model fit data (squared test & train error and num nonzero
    # coefficients) about different lambda values for chosen semantic features for both
    # SCD and PGD.

    # NOTE: LAMBDA values are different across the two
    # SET THE LAST PARAMETER TO FALSE TO PLOT MODEL FIT DATA USING SCD AND TRUE FOR PGD

    lambda_values_pgd = [.1, .5, 1, 5, 10, 20, 40, 100, 200]
    lambda_values_scd = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

    '''
    plot_semantic_feature_squared_error(signals_test, signals_train,
                                        words_test, words_train, semantic_features,
                                        lambda_values_scd, False)
   
    plot_semantic_feature_squared_error(signals_test, signals_train,
                                        words_test, words_train, semantic_features,
                                        lambda_values_pgd, True)
     '''
    y = np.zeros((len(semantic_features[0]), len(words_train)))
    y_test = np.zeros((len(semantic_features[0]), len(words_test)))
    model_weights = np.zeros((len(semantic_features[0]), len(signals_train[0])))

    # Build a linear model for every semantic feature
    for i in range(len(semantic_features[0])):
        print("Building linear model for semantic feature " + str(i + 1) + " :")
        # build the y-train vector for the current semantic feature
        for j in range(len(words_train)):
            word_index = words_train[j]
            y[i][j] = semantic_features[word_index - 1][i]

        # build the y-test vector for the current semantic feature
        for j in range(len(words_test)):
            word_index = words_test[j][0]
            # Word test stores things as floats, and the lookup in semantic features doesn't work
            # unless it is converted to int
            word_index = int(word_index)
            y_test[i][j] = semantic_features[word_index - 1][i]

        # Pull out the current y and y_test vectors
        cur_y = y[i]
        cur_y_test = y_test[i]

        # Initialize the KFold split generator
        # Performing 10 fold cross validation on training set
        num_folds = 10
        kf = KFold(n_splits=num_folds)
        min_lambda = lambda_values_scd[0]
        min_error = 0
        best_weights = np.zeros(len(signals_train[0]))

        # If using kfold check a range of lambda values and determine CV error on each one
        # to find min error and best lambda for this particular model
        # Otherwise use test set to determine best lambda choice
        if use_k_fold:
            for k in range(len(lambda_values_scd)):
                cross_validation_error = 0.0
                cur_lambda = lambda_values_scd[k]
                print("Testing lambda value: " + str(lambda_values_scd[k]))
                for index_train, index_valid in kf.split(signals_train):
                    weights = np.zeros(len(signals_train[0]))
                    X_train, X_valid = signals_train[index_train], signals_train[index_valid]
                    y_train, y_valid = cur_y[index_train], cur_y[index_valid]
                    weights = scd(cur_lambda, y_train, X_train, weights, 20)

                    cross_validation_error += squared_error(y_valid, X_valid, weights)
                avg_cv_error = cross_validation_error / float(num_folds)
                if k == 0 or avg_cv_error < min_error:
                    min_error = avg_cv_error
                    min_lambda = lambda_values_scd[k]
                    best_weights = weights
            model_weights[i] = best_weights
        else:
            for k in range(len(lambda_values_scd)):
                print("Testing lambda value: " + str(lambda_values_scd[k]))
                weights = np.zeros(len(signals_train[0]))
                cur_lambda = lambda_values_scd[k]
                weights = scd(cur_lambda, cur_y, signals_train, weights, 20)
                squared_error_test = squared_error(cur_y_test, signals_test, weights)
                print("Error for this lambda: " + str(squared_error_test))
                if k == 0 or squared_error_test < min_error:
                    min_error = squared_error_test
                    min_lambda = lambda_values_scd[k]
                    best_weights = weights
            model_weights[i] = best_weights

        # Print end results for the current model
        print("Best lambda: " + str(min_lambda))
        print("Error on this model: " + str(min_error))
        print("Number of nonzero coefficients " + str(np.count_nonzero(model_weights[i])))

    # All linear models have been built and
    # model_weights should now contain 218 linear models for
    # each of the semantic features
    print("All linear models built")
    scipy.io.mmwrite("model_weights.mtx", model_weights)


# Plots model fit data (squared test & train error and num nonzero
# coefficients) about different lambda values for chosen semantic features
def plot_semantic_feature_squared_error(signals_test, signals_train,
                                        words_test, words_train, semantic_features,
                                        lambda_values, is_pgd):

    # Initialize data maps and the semantic features that we will test
    semantic_features_to_test = [0, 99, 199]
    semantic_features_map_train = {0: [], 99: [], 199: []}
    semantic_features_map_test = {0: [], 99: [], 199: []}
    semantic_features_map_non_zero = {0: [], 99: [], 199: []}

    # Build the y component of the lasso algorithm for the test semantic features
    y = np.zeros((3, len(words_train)))
    for i in range(len(y)):
        for j in range(len(words_train)):
            cur_feature = semantic_features_to_test[i]
            word_index = words_train[j]
            y[i][j] = semantic_features[word_index - 1][cur_feature]

    # Build the y vector for test data
    y_test = np.zeros((3, len(words_test)))
    for i in range(len(y_test)):
        for j in range(len(words_test)):
            cur_feature = semantic_features_to_test[i]
            word_index = words_test[j][0]
            # Word test stores things as floats, and the lookup in semantic features doesn't work
            # unless it is converted to int
            word_index = int(word_index)
            y_test[i][j] = semantic_features[word_index - 1][cur_feature]

    # For each semantic feature we want to test we will either use
    # SCD (Stochastic Coordinate Descent) or PGD (Proximal Gradient Descent) to
    # generate models for different tuning parameters and collect data that we
    # will ultimately use to plot graphs
    for i in range(len(semantic_features_to_test)):
        for j in range(len(lambda_values)):
            weights = np.zeros(len(signals_train[0]))
            cur_lambda = lambda_values[j]
            cur_semantic_feature = semantic_features_to_test[i]
            cur_y = y[i]
            cur_y_test = y_test[i]
            # Based on the flag, we choose the appropriate method
            if is_pgd:
                weights = pgd(cur_lambda, cur_y, signals_train, weights, 20, 20)
            else:
                weights = scd(cur_lambda, cur_y, signals_train, weights, 30)

            # Collect performance metrics on the current model
            squared_error_train = squared_error(cur_y, signals_train, weights)
            squared_error_test = squared_error(cur_y_test, signals_test, weights)
            nonzero = np.count_nonzero(weights)

            # Add it to the data maps
            semantic_features_map_train[cur_semantic_feature].append((math.log(cur_lambda), squared_error_train))
            semantic_features_map_test[cur_semantic_feature].append((math.log(cur_lambda), squared_error_test))
            semantic_features_map_non_zero[cur_semantic_feature].append((math.log(cur_lambda), nonzero))

    # Define method and generate plot showing error and num nonzero coef
    method = "PGD" if is_pgd else "SCD"
    plot_squared_error(semantic_features_map_train, True, method)
    plot_squared_error(semantic_features_map_test, False, method)
    plot_num_zero(semantic_features_map_non_zero, method)


# Generates a plot that shows how changes
# in lambda affect the squared error of test and train
# data_map - map of semantic feature to map of lambda to squared error
# train_flag - True if plotting the training error, false if plotting test error
# method - The method used (PGD or SCD)
def plot_squared_error(data_map, train_flag, method):
    for key in data_map:
        li = data_map[key]
        log_lambda, data = zip(*li)
        plt.plot(log_lambda, data, label="Semantic Feature " + str(key + 1))
    plt.xlabel("ln(lambda)")
    plt.ylabel("Squared Error")
    plt.legend()
    if train_flag:
        plt.title("Log Lambda vs. Squared Error in Training Data (For " + method + ")")
        plt.savefig("squaredErrorTrain_" + method + ".png")
        plt.close()
    else:
        plt.title("Log Lambda vs. Squared Error in Test Data (For " + method + ")")
        plt.savefig("squaredErrorTest_" + method + ".png")
        plt.close()


# Generates a plot that shows how changes
# in tuning parameter lambda affect the number of
# nonzero coefficients in the weights vector
# data_map - map of semantic feature to map of lambda to  num nonzero coef
# method - The method used (PGD or SCD)
def plot_num_zero(data_map, method):
    for key in data_map:
        li = data_map[key]
        log_lambda, data = zip(*li)
        plt.plot(log_lambda, data, label="Semantic Feature " + str(key + 1))
    plt.xlabel("ln(lambda)")
    plt.ylabel("Num of Zero Coefficients")
    plt.legend()
    plt.title("Log of Lambda vs. Number of Zero Coefficients (For " + method + ")")
    plt.savefig("numNonZero_" + method + ".png")
    plt.close()


# Reads in the words train file into a vector
def import_words_train(file_name, array):
    f = open(file_name)
    index = 0
    for line in f:
        temp = line
        array[index] = int((temp.split())[0])
        index += 1
    return array


# Returns the squared error value for
# a given model
def squared_error(y, X, weights):
    y2 = np.copy(y)
    for i in range(X.shape[0]):
        y2[i] = np.dot(X[i], weights)

    diff = y-y2
    sum_error = np.sum(np.square(diff))
    return sum_error / X.shape[0]


if __name__ == "__main__":
    main()
