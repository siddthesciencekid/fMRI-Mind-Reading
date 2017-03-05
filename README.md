# fMRI-Mind-Reading
Final Project for the Machine Learning Class (CSE 446) at the University of Washington

# Abstract
This project involves discovering how fMRI brain scans can be used to predict what word a person is reading based off of activation patterns in the brain. The goal is to first learn 218 sparse linear models, each predicting a semantic feature of the word based on an fMRI input. Using Coordinate Descent and Stochastic Coordinate Descent for Lasso, as well Proximal Gradient Descent with L1 penalty, we figured out the coefficients to predict the value of a semantic feature. Given a new brain scan input,  we are able to build a 218-dimensional vector representing the values of semantic features of the word read. Given two candidate words, we use binary classification to choose the word whose semantic feature vector is closer to the predicted one.


## Project Contributers
1. Shilpa Kumar
2. Siddhartha Gorti

## Data Files
The data files for this project can be found at this link: http://www.cs.cmu.edu/afs/cs/project/theo-73/www/science2008/data.html

## Model Weights
The generated model weights file is very large (>100 MB) and thus cannot be effectively tracked by git.
Therefore the model weights file has been compressed and added to this repository. When working with the fmri_solver, please
unzip the model_weights file or generate it to use the program

# USAGE INSTRUCTIONS:
`fmri_solver.py <function> <optional_param>`

## FUNCTION LIST:
### test_algs
Use this to plot model fit data (squared test & train error and num nonzero
coefficients) about different lambda values for chosen semantic features for both
SCD and PGD.

### build_model
Use this to build the 218 x 21764 matrix where each row i is a linear model
to generate semantic feature i from the 21764 voxel features given from a brain scan. The models
are saved in the local directory as an mtx file "model_weights.mtx"

NOTE: build_model requires `<optional param>`
	
1. kfold: To use kfold cross validation to choose the best tuning parameter

  * WARNING: ENABLING KFOLD CROSS VALIDATION SIGNIFICANTLY INCREASES THE RUN TIME OF THIS PROGRAM

2. test_set: To use test set validation to choose the best tuning parameter

### test_model
Use this to test the model against our test data. For a given brain scan input we use
the model weights to generate 218 x 1 vector and use 1-NN classification to determine between two words
which word was most likely read. Plot information about the model's mistake rate when given a set of known
test words and associated brain scan will be generated.

NOTE: Requires `model_weights.mtx` file to be present. Use the build_model function if lacking or download from 
[our Github](https://github.com/siddthesciencekid/fMRI-Mind-Reading/blob/master/model_weights.zip "Model Weights Download").

Example usage: `fmri_solver.py build_model kfold` & 
`fmri_solver.py test_model`