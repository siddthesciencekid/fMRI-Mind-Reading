# Siddhartha Gorti & Shilpa Kumar
# FMRI Project
# CSE 446 Machine Learning
# WINTER 2017

import scipy.io
import numpy as np

def import_words_train(fileName, array):
	f = open(fileName)
	index = 0
	for line in f:
		temp = line
		array[index] = int((temp.split())[0])
		index+=1
	return array


def main():
    signals_test = scipy.io.mmread("data/subject1_fmri_std.test.mtx")
    signals_train = scipy.io.mmread("data/subject1_fmri_std.train.mtx")
    words_test = scipy.io.mmread("data/subject1_wordid.test.mtx")

    #Training wordid only has 1 column, and using mmread doesn't work
    words_train = [0 for x in range(300)]
    words_train = import_words_train("data/subject1_wordid.train.mtx", words_train)
    words_train = np.asarray(words_train)
    
    semanticFeatures = scipy.io.mmread("data/word_feature_centered.mtx")

    print(signals_train.shape)
    print(signals_test.shape)
    print(words_train.shape)
    print(words_test.shape)
    print(semanticFeatures.shape)

if __name__ == "__main__":
    main()
