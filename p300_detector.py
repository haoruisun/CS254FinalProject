#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 15:24:23 2022

@author: hsun11
"""

# %% Import packages
import numpy as np
import loadmat as lm
from matplotlib import pyplot as plt
from scipy import signal
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.utils import resample


# %% Proprocess func
def preprocess(data_dict, ch_index, fs, epoch_period=667, downsample_size=2500):
    '''
    This function loads the data struct to extract machine learning features 
    and downsample. 

    Parameters
    ----------
    data_dict : dict
        DESCRIPTION. The EEG data dictionary
    ch_index : int
        DESCRIPTION. The integer of desired channel
    fs : int
        DESCRIPTION. Sampling frequency in Hz
    epoch_period : int, optional
        DESCRIPTION. The epoch duration in ms. 
                     The default is 667.
    downsample_size : int, optional
        DESCRIPTION. The downsampled size.
                     The default is 2500.

    Returns
    -------
    X : 2d array: sample x feature
        DESCRIPTION. Training data
    y : 1d array
        DESCRIPTION. Labels

    '''
    # extract info
    data = data_dict['Signal'][:, :, ch_index]
    flashing = data_dict['Flashing']
    types= data_dict['StimulusType']
    
    # get the event index
    event_time = np.where(np.diff([flashing])[0][0] > 0)[0] + 1
    # manually insert the first event at index 0
    event_time = np.insert(event_time, 0, 0)
    # extract labels for each sample/epoch
    label = types[:, event_time].flatten()
    
    # calculate the period length for each sample
    epoch_len = round(epoch_period * fs / 1000)
    # declare the 3d epoch array
    epochs = np.zeros((data.shape[0], len(event_time), epoch_len))
    # loop through the eeg data into epochs
    for epoch_index, epoch_start in enumerate(event_time):
        epoch_end = epoch_start + epoch_len
        epochs[:, epoch_index] = data[:, epoch_start:epoch_end]
    # reshape the epochs array to 2d
    sample = epochs.reshape((epochs.shape[0]*epochs.shape[1]), epochs.shape[2])
    
    # create a 8-order bandpass Chebyshev Type I filter which cut-off 
    # frequencies are 0.1 and 10 Hz
    filter_coefficients = signal.firwin(8, [0.5, 10], window='hann', 
                                        pass_zero=False, fs=fs)
    # filter the eeg data
    filtered_sample = signal.filtfilt(filter_coefficients, 1, sample)
    
    # downsample
    # form the dataset
    dataset = np.hstack((filtered_sample, np.reshape(label, (-1, 1))))
    # extract target and nontarget epochs
    target = dataset[np.where(label==1)[0]]
    nontarget = dataset[np.where(label==0)[0]]
    # use resample to downsample 
    target_ds =  resample(target, replace=False, n_samples=downsample_size)
    nontarget_ds =  resample(nontarget, replace=False, n_samples=downsample_size)
    # merge two classes
    dataset_ds = np.vstack((target_ds, nontarget_ds))
    
    # return values
    X = dataset_ds[:, :-1]
    y = dataset_ds[:, -1]
    return X, y


# %% SVM func
def svm(X, y):
    '''
    This function trains a SVM classifer on given data and evaluates its 
    performance by plotting the confusion matrix.

    X : 2d array: sample x feature
        DESCRIPTION. Training data
    y : 1d array
        DESCRIPTION. Labels

    Returns
    -------
    clf_svm : SVM object
        DESCRIPTION. The trained SVM classifer

    '''
    # split the data into training and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    # scale datasets
    X_train_scaled = scale(X_train)
    X_test_scaled = scale(X_test)
    
    # optimize parameters with cross validation
    # set up potential parameters
    print('Optimize parameters with cross validation...')
    param_grid = [
        {'C': [0.01, 0.05, 0.1, 0.5, 1],
         'kernel': ['linear']}
        ]
    # set up cross validation
    optimal_params = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy')
    # compute the optimal parameters
    optimal_params.fit(X_train_scaled, y_train)
    # print the optimal parameters
    print('The optimal parameters are:', optimal_params.best_params_)
    # lock the values for C and gamma
    C = optimal_params.best_params_['C']
    
    # build the final svm
    clf_svm = SVC(C=C, kernel='linear')
    # train the model
    print('Train the model...')
    clf_svm.fit(X_train_scaled, y_train)
    print('Now the classifier has been trained.')
    
    # generate the predicted labels
    print('Predict labels for tested dataset...')
    y_pred = clf_svm.predict(X_test_scaled)
    # calculate the accuracy score
    accuracy = accuracy_score(y_test, y_pred)
    # print the results
    print('Accuracy: ', accuracy)
    
    # plot the confusion matrix
    print('Plot the confusion matrix...')
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.show()
    plt.text(4.3, 2.1, 'number of samples', rotation=90)
    plt.title('SVM Confusion Matrix')
    plt.savefig('svm_confusion_matrix.png')
    # display the main classification metrics
    print('Display the main classification metrics:')
    print(classification_report(y_test, y_pred))
    return clf_svm
    

# %% P300 detector
# define data files 
file = 'Subject_A_Train.mat'
# declare the experiment parameters
fs = 240        # sampling frequency in Hz
channel = 57    # POz channel index

# load the training data
data_dict = lm.loadmat(file)
# proprocess data
X, y = preprocess(data_dict, channel, fs)
# train the svm classifer
clf_svm = svm(X, y)



















