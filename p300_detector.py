#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 15:24:23 2022
Modified on Tue May 3 -- add plotting function
                      -- add regression classifer

@author: Haorui Sun and Claire Davis
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
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import optimizers

# %% Helper func
def preprocess(data_dict, ch_index, fs, epoch_period=667, downsample_size=1000):
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
    # reform the dataset
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


def plot_data(X, y, title, fs, 
              xlabel='Time After Stimulus (ms)', 
              ylabel='Signal Amplitude (A/D Units)'):
    '''
    This function visualizes splitted training data for mannual inspection.
    Plot will be saved to current working directory.

    Parameters
    ----------
    X : 2d array: sample x feature
        DESCRIPTION. Training data
    y : 1d array
        DESCRIPTION. Labels
    title : string
        DESCRIPTION. The plot's title and name
    fs : int
        DESCRIPTION. Sampling frequency in Hz
    xlabel : string, optional
        DESCRIPTION. The label for x-axis.
                    The default is 'Time After Stimulus (ms)'.
    ylabel : string, optional
        DESCRIPTION. The label for y-axis.
                    The default is 'Signal Amplitude (A/D Units)'.

    Returns
    -------
    None.

    '''
    
    # split training data into target and non-target and take average
    target = np.mean(X[np.where(y==1)[0]], axis=0)
    nontarget = np.mean(X[np.where(y==0)[0]], axis=0)
    # convert x-axis to time series
    time = np.arange(len(target))/fs*1000
    
    # plot the data
    plt.figure()
    plt.plot(time, target, label='target')
    plt.plot(time, nontarget, label='nontarget')
    # annotate the plot
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(title+'.png')
    

def plot_matrix(X, y, model, label):
    '''
    This function takes trained model and testing data, and prints out its
    perfermance on the testing data.

    Parameters
    ----------
    X : 2d array: sample x feature
        DESCRIPTION. Testing data
    y : 1d array
        DESCRIPTION. Testing labels
    model : ML object
        DESCRIPTION. Trained classifer
    label : string
        DESCRIPTION. The label for confusion matrix

    Returns
    -------
    None.

    '''
    # generate the predicted labels
    print('Predict labels for tested dataset...')
    y_pred = model.predict(X)
    # calculate the accuracy score
    accuracy = accuracy_score(y, y_pred)
    # print the results
    print('Accuracy: ', accuracy)
    
    # plot the confusion matrix
    print('Plot the confusion matrix...')
    ConfusionMatrixDisplay.from_predictions(y, y_pred)
    plt.show()
    plt.text(4.3, 2.1, 'number of samples', rotation=90)
    plt.title(label)
    plt.savefig(label+'.png')
    # display the main classification metrics
    print('Display the main classification metrics:')
    print(classification_report(y, y_pred))


# %% Model func
def log_reg(X, y):
    
    # split the data into training and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(scale(X), y)
    print('Train Logistic Regression model...')
    clf_lg = LogisticRegression(random_state=0).fit(X_train, y_train)
    print('Now the classifier has been trained.')
    
    # print report and plot confusion matrix
    label = 'LogReg_confusion_matrix'
    plot_matrix(X_test, y_test, clf_lg, label)
    # return trained classifier
    return clf_lg



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
    print('Train SVM model...')
    clf_svm.fit(X_train_scaled, y_train)
    print('Now the classifier has been trained.')
    
    # print report and plot confusion matrix
    label = 'svm_confusion_matrix'
    plot_matrix(X_test_scaled, y_test, clf_svm, label)
    # return trained classifier
    return clf_svm
    

def dnn(X, y, optimizer=optimizers.Adam(), loss='mse'):
    
    # split the data into training, validating, and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    
    # design neural network architecture
    layers = [
        Flatten(),
        Dense(32, activation="relu"),
        Dense(8, activation="relu"),
        Dense(1)
    ]
    dnn = Sequential(layers)
    # compile neural network
    dnn.compile(optimizer=optimizer, loss=loss, 
                  metrics=['accuracy'])
    # train the network and save each epoch output in the history list
    history = dnn.fit(X_train, y_train, batch_size=16, epochs=10, 
                        validation_split=0.2, verbose=0, callbacks=[])
    # plot the performance
    plt.figure()
    plt.plot(history.history['accuracy'], label="acc")
    plt.plot(history.history['val_accuracy'], label="val_acc")
    plt.legend()
    plt.show()
    
    # return trained network
    return dnn


# %% P300 detector
# define data files 
file = 'Subject_A_Train.mat'
# declare the experiment parameters
fs = 240        # sampling frequency in Hz
channel = 10    # POz channel index

# load the training data
data_dict = lm.loadmat(file)
# proprocess data
X, y = preprocess(data_dict, channel, fs)
# plot the training data
plot_data(X, y, 'SA_training_data', fs)



# %% Test log_reg
clf_lg = log_reg(X, y)

# %% train the svm classifer
clf_svm = svm(X, y)

# %% Test dnn 
dnn = dnn(X, y)



