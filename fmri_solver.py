# Siddhartha Gorti & Shilpa Kumar
# FMRI Project
# CSE 446 Machine Learning
# WINTER 2017

import scipy.io

def main():
    signalsTest = scipy.io.mmread("data/subject1_fmri_std.test.mtx")
    signalsTrain = scipy.io.mmread("data/subject1_fmri_std.train.mtx")
    #train not working
    print("hello???")
    wordsTest = scipy.io.mmread("data/subject1_wordid.test.mtx")
    wordsTrain = scipy.io.mmread("data/subject1_wordid.train.mtx")
    semanticFeatures = scipy.io.mmread("data/word_feature_centered.mtx")
    print(wordsTest.shape)

if __name__ == "__main__":
    main()
