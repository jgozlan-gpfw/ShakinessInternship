import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from numpy import array, hstack
import random


def create_X_Y(ts: np.array, lag=1, n_ahead=1, delay = 1, target_index= [0,1,2], classification=False) -> tuple:
    """
    A method to create X and Y matrix from a time series array for the training of 
    deep learning models 
    """
    # Extracting the number of features that are passed from the array
    
    # Creating placeholder lists
    X, Y = [], []

    if len(ts) - lag <= 0:
        print("too small")
    else:
        if classification:
            for i in range(0, len(ts) - lag - n_ahead, delay):
                Y.append(ts[i][target_index])
                X.append(ts[i:(i + lag), :-1])
            #x.append(ts[(i + lag):(i + lag + n_ahead), target_index]) 
            #current_x = np.append(ts[i:(i + lag), : -1], np.ones((lag,1)) * calib, axis = 1)
            #X.append(current_x)
        else:
            for i in range(0, len(ts) - lag - n_ahead, delay):
                Y.append(ts[(i + lag):(i + lag + n_ahead), target_index])
                X.append(ts[i:(i + lag)])

    X, Y = np.array(X), np.array(Y)
    
    n_features = ts.shape[1]

    if classification:
        X = np.reshape(X, (X.shape[0], lag, n_features -1))
    else:
        X = np.reshape(X, (X.shape[0], lag, n_features))

    return X, Y


def get_df_chunked_train_test(df, test_size, lag = 500, n_ahead = 50, delay =1, target_index = [0,1,2], classification = False):
    
    chunk_size = int(test_size * df.shape[0])
    
    X_trains = []
    y_trains =[]
    X_tests = []
    y_tests = []
    
    
    if (lag + n_ahead) <= chunk_size :
        
        nb_full_chunks  = int(df.shape[0] / chunk_size)
        #print("Nb full chunks:", nb_full_chunks)
        
        test_index = random.randint(1, nb_full_chunks - 2)
        
        train_chunk0 = df.iloc[0:chunk_size * test_index]
        X_train0, y_train0 = create_X_Y(train_chunk0.to_numpy(), lag, n_ahead, delay, target_index, classification)
        if len(X_train0) > 0:
            X_trains.append(X_train0)
            y_trains.append(y_train0)
                
        
        test_chunk = df.iloc[chunk_size * test_index : chunk_size * (test_index+1)]
        X_test, y_test = create_X_Y(test_chunk.to_numpy(), lag, n_ahead, delay, target_index, classification)
        if len(X_test) > 0:
            X_tests.append(X_test)
            y_tests.append(y_test)
                
        
        train_chunk1 = df.iloc[chunk_size * (test_index+1):]
        X_train1, y_train1 = create_X_Y(train_chunk1.to_numpy(), lag, n_ahead, delay, target_index, classification)
        if len(X_train1) > 0:
            X_trains.append(X_train1)
            y_trains.append(y_train1)

    return X_trains, y_trains, X_tests, y_tests


def get_train_test_split(dfs, test_size, lag = 1000, n_ahead = 100, delay =1, target_index = [0,1,2], classification = False):

    all_X_train = []
    all_y_train = []
    all_X_test = []
    all_y_test = []
    
    for df in dfs:

        chunk_X_train, chunk_y_train, chunk_X_test, chunk_y_test = get_df_chunked_train_test(df, test_size, lag, n_ahead, delay, target_index, classification)
        all_X_train.append(chunk_X_train)
        all_y_train.append(chunk_y_train)
        all_X_test.append(chunk_X_test)
        all_y_test.append(chunk_y_test)
    
    X_train = [item for sublist in all_X_train for item in sublist]
    y_train = [item for sublist in all_y_train for item in sublist]
    X_test = [item for sublist in all_X_test for item in sublist]
    y_test = [item for sublist in all_y_test for item in sublist]
     
        # we have a list of list => stack apres reduit en 1 seule liste
        
    X_train = np.vstack(X_train)
    X_test = np.vstack(X_test)

    if classification:
        y_train = np.concatenate(y_train, axis=0)
        y_test = np.concatenate(y_test, axis=0)
    else:
        y_train = np.vstack(y_train)
        y_test = np.vstack(y_test)

        
    return X_train, y_train, X_test, y_test

  
   