# Siddhartha Gorti & Shilpa Kumar
# Final Project
# CSE 446 Machine Learning
# WINTER 2017

import scipy.io
import numpy as np
import timeit
import math

from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from lasso import lasso
from scd import scd
from pgd import pgd


def main():
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

    '''
    lambda_values_pgd = [.1, .5, 1, 5, 10, 20, 40, 100, 200]
    lambda_values_scd = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, .4]


    plot_semantic_feature_squared_error(signals_test, signals_train,
                                        words_test, words_train, semantic_features,
                                        lambda_values_scd, False)

    plot_semantic_feature_squared_error(signals_test, signals_train,
                                        words_test, words_train, semantic_features,
                                        lambda_values_pgd, True)
    '''

    # TODO: START CROSS VALIDATION FOR ONE MODEL
    # Performing 5 fold cross validation
    K = 5

    y = np.zeros((len(words_train), 1))
    for i in range(len(words_train)):
        word_index = words_train[i]
        y[i][0] = semantic_features[word_index - 1][0]



def k_fold_generator(X, y, k_fold):
    subset_size = int(len(X) / k_fold)
    for k in range(k_fold):
        X_train = X[:k * subset_size] + X[(k + 1) * subset_size:]
        X_valid = X[k * subset_size:][:subset_size]
        y_train = y[:k * subset_size] + y[(k + 1) * subset_size:]
        y_valid = y[k * subset_size:][:subset_size]

        return X_train, y_train, X_valid, y_valid

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
