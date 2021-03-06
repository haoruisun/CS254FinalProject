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
import math
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression
from tensorflow import keras
from tensorflow.keras.models import Sequential 
from tensorflow.keras import optimizers
from tensorflow.keras import utils as np_utils
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Permute, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.constraints import max_norm
import mne
from mne import io
from mne.datasets import sample


# %% Helper func
def preprocess(data_dict, ch_indices, fs, epoch_period=667):
    '''
    This function loads the data struct and extracts machine learning features
    from raw dataset. 

    Parameters
    ----------
    data_dict : dict
        DESCRIPTION. The EEG data dictionary
    ch_indices : int list/array
        DESCRIPTION. The list/array of the integer of desired channel
    fs : int
        DESCRIPTION. Sampling frequency in Hz
    epoch_period : int, optional
        DESCRIPTION. The epoch duration in ms. 
                     The default is 667.

    Returns
    -------
    X_scaled : 2d array: sample x feature
        DESCRIPTION. Scaled training data
    y : 1d array
        DESCRIPTION. Labels

    '''
    # extract info
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
    
    # extract data for each electode
    sample_all_electrodes = None
    for ch in ch_indices:
        data = data_dict['Signal'][:, :, ch]
        # declare the 3d epoch array
        epochs = np.zeros((data.shape[0], len(event_time), epoch_len))
        # loop through the eeg data into epochs
        for epoch_index, epoch_start in enumerate(event_time):
            epoch_end = epoch_start + epoch_len
            epochs[:, epoch_index] = data[:, epoch_start:epoch_end]
        # reshape the epochs array to 2d
        sample = epochs.reshape((epochs.shape[0]*epochs.shape[1]), 
                                epochs.shape[2])
        
        #--------------extract 14 samples---------------------
        indices = np.arange(0, epochs.shape[2], int(epochs.shape[2]/14))
        sample = sample[:, indices]
        
        # combine electrode feature(s)
        if sample_all_electrodes is None:
            sample_all_electrodes = sample
        else:
            # sample_all_electrodes = sample_all_electrodes + sample
            sample_all_electrodes = np.hstack((sample_all_electrodes, sample))
    
    
    
    # ------------------ Filter -----------------------
    # create a 8-order bandpass Chebyshev Type I filter which cut-off 
    # frequencies are 0.1 and 10 Hz
    # filter_coefficients = signal.firwin(8, [0.5, 10], window='hann', 
    #                                     pass_zero=False, fs=fs)
    # # filter the eeg data
    # filtered_sample = signal.filtfilt(filter_coefficients, 1, sample)
    # # reform the dataset
    # dataset = np.hstack((filtered_sample, np.reshape(label, (-1, 1))))
    
    # testing...
    #------------
    #X = sample_all_electrodes/len(ch_indices)
    X = sample_all_electrodes
    y = label
    #------------
    
    # normalize X
    scaler = preprocessing.StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    
    return X_scaled, y


def preprocess1(data_dict, ch_indices, fs, epoch_period=667):
    '''
    This function loads the data struct and extracts machine learning features
    for EEGNet only from raw dataset. 

    Parameters
    ----------
    data_dict : dict
        DESCRIPTION. The EEG data dictionary
    ch_indices : int list/array
        DESCRIPTION. The list/array of the integer of desired channel
    fs : int
        DESCRIPTION. Sampling frequency in Hz
    epoch_period : int, optional
        DESCRIPTION. The epoch duration in ms. 
                     The default is 667.

    Returns
    -------
    X_ds : 2d array: sample x feature
        DESCRIPTION. Downsampled training data
    y_ds : 1d array
        DESCRIPTION. Labels

    '''
    # extract info
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
    
    # extract data for each electode
    X = []
    for ch in ch_indices:
        data = data_dict['Signal'][:, :, ch]
        # declare the 3d epoch array
        epochs = np.zeros((data.shape[0], len(event_time), epoch_len))
        # loop through the eeg data into epochs
        for epoch_index, epoch_start in enumerate(event_time):
            epoch_end = epoch_start + epoch_len
            epochs[:, epoch_index] = data[:, epoch_start:epoch_end]
        # reshape the epochs array to 2d
        sample = epochs.reshape((epochs.shape[0]*epochs.shape[1]), 
                                epochs.shape[2])
        # combine all data
        X.append(sample)
        
    # reshape X
    X = np.asarray(X)
    X = X.reshape(X.shape[1], X.shape[0], X.shape[2])
    
    
    # downsample size
    y= label
    downsample_size = 250
    # split target and nontarget samples
    target = X[np.where(y==1)[0]]
    nontarget = X[np.where(y==0)[0]]
    # generate indices
    target_ind = resample(np.arange(target.shape[0]), 
                          replace=False, n_samples=downsample_size)
    nontarget_ind = resample(np.arange(nontarget.shape[0]), 
                             replace=False, n_samples=downsample_size)
    # merge two classes
    X_ds = np.vstack((target[target_ind], nontarget[nontarget_ind]))
    y_ds = np.vstack((np.ones((downsample_size, 1)), 
                      np.zeros((downsample_size, 1))))
    
    return X_ds, y_ds



def downsample(X, y, downsample_size=2000):
    '''
    This function downsample-size training data.

    Parameters
    ----------
    X : 2d array: sample x feature
        DESCRIPTION. Testing data
    y : 1d array
        DESCRIPTION. Testing labels
    downsample_size : int, optional
        DESCRIPTION. The downsampled data size 
        The default is 2000.

    Returns
    -------
    X_ds : 2d array: sample x feature
        DESCRIPTION. Downsampled training data
    y_ds : 1d array
        DESCRIPTION. Labels

    '''
    
    # split target and nontarget samples
    target = X[np.where(y==1)[0]]
    nontarget = X[np.where(y==0)[0]]
    # use resample to downsample 
    target_ds =  resample(target, replace=False, n_samples=downsample_size)
    nontarget_ds = resample(nontarget, replace=False, n_samples=downsample_size)
    # merge two classes
    X_ds = np.vstack((target_ds, nontarget_ds))
    y_ds = np.vstack((np.ones((downsample_size, 1)), 
                      np.zeros((downsample_size, 1))))
    
    # return downsample-sized data
    return X_ds, y_ds


def plot_erp(X, y, title, fs, 
             plot_ci=False,
             xlabel='Time After Stimulus (ms)', 
             ylabel='Signal Amplitude (uV)'):
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
    
    # split training data into target and non-target
    target = X[np.where(y==1)[0]]
    nontarget = X[np.where(y==0)[0]]
    # compute the average
    target_mean = np.mean(target, axis=0)
    nontarget_mean = np.mean(nontarget, axis=0)
    # compute the std across trials
    target_sd = np.std(target, axis=0)
    nontarget_sd = np.std(nontarget, axis=0)
    # compute the std of the mean
    target_sdmn = target_sd / math.sqrt(len(target_mean)) # number of epochs
    nontarget_sdmn = nontarget_sd / math.sqrt(len(nontarget_mean))
    # convert x-axis to time series
    time = np.arange(len(target_mean))/fs*1000
    
    # plot erp mean
    plt.figure()
    plt.plot(time, target_mean, color='deepskyblue', label='target')
    plt.plot(time, nontarget_mean, color='orange', label='nontarget')
    # plot erp 95 CI
    if plot_ci:
        plt.fill_between(time, 
                          target_mean - 2 * target_sdmn,
                          target_mean + 2 * target_sdmn,
                          color='deepskyblue', alpha=0.2,
                          label='Target +/- 95% CI')
        plt.fill_between(time, 
                          nontarget_mean - 2 * nontarget_sdmn,
                          nontarget_mean + 2 * nontarget_sdmn,
                          color='orange', alpha=0.2,
                          label='Nontarget +/- 95% CI')
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
    # ====================================================
    # if for neural network
    # y_pred = model.predict_classes(X)
    # ====================================================
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
    '''
    This function trains a logistic regression classifer on given data and 
    evaluates its performance by plotting the confusion matrix.

    Parameters
    ----------
    X : 2d array: sample x feature
        DESCRIPTION. Training data
    y : 1d array
        DESCRIPTION. Labels

    Returns
    -------
    clf_lg : logistic regression object
        DESCRIPTION. The trained LG classifer

    '''
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
    
    Parameters
    ----------
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
        {'C': [0.001, 0.01, 0.05, 0.1, 0.5, 1, 10, 100],
         'kernel': ['rbf']}
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
    

def dnn(X, y, optimizer=optimizers.Adam(learning_rate=0.001), loss='mse'):
    '''
    This function trains a neural network on given data.

    Parameters
    ----------
    X : 2d array: sample x feature
        DESCRIPTION. Training data
    y : 1d array
        DESCRIPTION. Labels
    optimizer : optimizer object, optional
        DESCRIPTION. The optimizer for neural network
        The default is optimizers.Adam().
    loss : string, optional
        DESCRIPTION. The loss functoin for neural network
        The default is 'mse'.

    Returns
    -------
    history : network object
        DESCRIPTION. The trained network information

    '''
    # split the data into training, validating, and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    
    # design neural network architecture
    layers = [
        Flatten(),
        Dense(70, activation='relu', 
              kernel_regularizer=keras.regularizers.l1(l=0.01)),
        Dense(10, activation='relu'),
        Dense(1, activation='sigmoid')
    ]
    dnn = Sequential(layers)
    # compile neural network
    dnn.compile(optimizer=optimizer, loss=loss, 
                  metrics=['accuracy'])
    # train the network and save each epoch output in the history list
    history = dnn.fit(X_train, y_train, batch_size=16, epochs=200, 
                      validation_split=0.2, verbose=0, callbacks=[])
    # plot the performance
    label = 'neural_network'
    plt.figure()
    plt.plot(history.history['accuracy'], label="acc")
    plt.plot(history.history['val_accuracy'], label="val_acc")
    plt.legend()
    plt.title(label)
    plt.savefig(label+'.png')
    
    label='nn_confusion_matrix'
    plot_matrix(X_test, y_test, dnn, label)
    
    # return trained network info
    return history


def cnn(X, y, fs, dropoutRate=0.5, F1=8, D=2, num_class=2,
        F2=16, norm_rate=0.25, dropoutType='Dropout', label='local_data'):
    '''
    This function designs a CNN based on EEGNet architecture. 

    Parameters
    ----------
    X : 2d array: sample x feature
        DESCRIPTION. Training data
    y : 1d array
        DESCRIPTION. Labels
    fs : int
        DESCRIPTION. sampling frequency
    dropoutRate : double, optional
        DESCRIPTION. The dropout rate
        The default is 0.5.
    F1 : int, optional
        DESCRIPTION. The number of the first filter
        The default is 8.
    D : int, optional
        DESCRIPTION. The default is 2.
    num_class : int, optional
        DESCRIPTION. The number of classes in the given dataset
        The default is 2.
    F2 : int, optional
        DESCRIPTION. The number of the second filter
        The default is 16.
    norm_rate : double, optional
        DESCRIPTION. Normalization rate
        The default is 0.25.
    dropoutType : string, optional
        DESCRIPTION. dropout method
        The default is 'Dropout'.
    label : string, optional
        DESCRIPTION. The title and name of saved plots
        The default is 'local_data'.

    Returns
    -------
    history : network object
        DESCRIPTION. The trained network information

    '''
    # define kernal length, which is good to be half of the sampling frequency
    kernLength = int(fs/2)
    # extract chans and samples info
    Chans = X.shape[1]
    Samples = X.shape[2]
    
    # define input size
    input1 = Input(shape=(Chans, Samples, 1))
    if dropoutType == 'Dropout':
        dropoutType = Dropout
    
    # ====================== MODEL ========================
    # Citation: 
    # https://github.com/vlawhern/arl-eegmodels/blob/master/EEGModels.py
    
    block1 = Conv2D(F1, (1, kernLength), padding = 'same',
                    input_shape = (Chans, Samples, 1),
                    use_bias = False)(input1)
    block1 = BatchNormalization()(block1)
    block1 = DepthwiseConv2D((Chans, 1), use_bias = False, 
                             depth_multiplier = D,
                             depthwise_constraint = max_norm(1.))(block1)
    block1 = BatchNormalization()(block1)
    block1 = Activation('elu')(block1)
    block1 = AveragePooling2D((1, 4))(block1)
    block1 = dropoutType(dropoutRate)(block1)
    
    block2 = SeparableConv2D(F2, (1, 16),
                             use_bias = False, padding = 'same')(block1)
    block2 = BatchNormalization()(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D((1, 8))(block2)
    block2 = dropoutType(dropoutRate)(block2)
        
    flatten = Flatten(name = 'flatten')(block2)
    
    dense = Dense(num_class, name = 'dense', 
                  kernel_constraint = max_norm(norm_rate))(flatten)
    softmax = Activation('softmax', name = 'softmax')(dense)
    
    # ====================================================
    model = Model(inputs=input1, outputs=softmax)
    
    # compile the network and set the optimizers
    model.compile(loss='categorical_crossentropy', optimizer='adam', 
                  metrics = ['accuracy'])
    
    # reshape X size
    X = X.reshape(X.shape[0], Chans, Samples, 1)
    # split train/test data
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    # train the network and save each epoch output in the history list
    history = model.fit(X_train, y_train, batch_size=16, epochs=100, 
                        validation_split=0.2, verbose=2)
    
    # plot the performance
    plt.figure()
    plt.plot(history.history['accuracy'], label="acc")
    plt.plot(history.history['val_accuracy'], label="val_acc")
    plt.legend()
    plt.title('EEGNet ' + label)
    plt.savefig(label + '.png')
    
    # return trained network info
    return history

# %% P300 detector
# define data files 
file = 'Subject_A_Train.mat'
# declare the experiment parameters
fs = 240        # sampling frequency in Hz
channel = [2, 3, 4, 5, 6, 10, 14, 15, 50, 55, 57, 59]    # POz channel index

# load the training data
data_dict = lm.loadmat(file)

# extract info
data = data_dict['Signal'][:, :, 10]
flashing = data_dict['Flashing']
types= data_dict['StimulusType']

# proprocess data
X, y = preprocess(data_dict, channel, fs)
# plot the training data
plot_erp(X, y, 'SA_training_data', fs, plot_ci=False, xlabel='temporal space')

# downsample the data
x_ds, y_ds = downsample(X, y, downsample_size=1000)
y_ds = y_ds.flatten()

# %% test log_reg
clf_lg = log_reg(x_ds, y_ds)

# %% test svm
clf_svm = svm(x_ds, y_ds)

# %% test dnn 
dnn = dnn(x_ds, y_ds)

# %% test EEGNet on local data
# proprocess data
channel = np.arange(64)
X1, y1 = preprocess1(data_dict, channel, fs)
# convert y to one-hot encodings
y1 = np_utils.to_categorical(y1)

# train the model
cnn = cnn(X1, y1, fs=fs)

# %% test EEGNet on online data (MNE data)
##################### Process, filter and epoch the data ######################
# Citation: 
#  https://github.com/vlawhern/arl-eegmodels/blob/master/examples/ERP.py
data_path = sample.data_path()

# Set parameters and read data
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'
tmin, tmax = -0., 1
event_id = dict(aud_l=1, aud_r=2, vis_l=3, vis_r=4)

# Setup for reading the raw data
raw = io.Raw(raw_fname, preload=True, verbose=False)
raw.filter(2, None, method='iir')  # replace baselining with high-pass
events = mne.read_events(event_fname)

raw.info['bads'] = ['MEG 2443']  # set bad channels
picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                       exclude='bads')

# Read epochs
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=False,
                    picks=picks, baseline=None, preload=True, verbose=False)
labels = epochs.events[:, -1]

# extract raw data. scale by 1000 due to scaling sensitivity in deep learning
X = epochs.get_data()*1000 # format is in (trials, channels, samples)
y = labels
# convert y to one-hot encodings
y = np_utils.to_categorical(y-1)

# train the EEGNet on MNE dataset
cnn = cnn(X, y, fs=64, num_class=4, label='mne_data')

# %% test dnn on MNE data
dnn = dnn(X, y, loss='categorical_crossentropy')