import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from numpy import array, hstack

import keras
from keras.models import Input, Model, Sequential

from keras.layers import Activation, Dense, Conv1D, MaxPooling1D, Dropout, LSTM, Concatenate, SimpleRNN, Masking, Flatten, TimeDistributed, RepeatVector
from keras import losses
from keras.callbacks import EarlyStopping
from keras.initializers import RandomNormal
from keras.optimizers import adam
from keras import backend as K

def create_lstm_encoder_decoder(encoder_n = 50, decoder_n = 50, lag = 500, ahead = 100, in_dim = 12, out_dim = 1):

    encoder_inputs = Input(shape=(lag, in_dim),name= "input1")
    encoder_l1 = LSTM(encoder_n,return_sequences = True, return_state=True, name= "lstm1")
    encoder_outputs1 = encoder_l1(encoder_inputs)
    encoder_states1 = encoder_outputs1[1:]
    encoder_l2 = LSTM(encoder_n, return_state=True, name= "lstm2")
    encoder_outputs2 = encoder_l2(encoder_outputs1[0])
    encoder_states2 = encoder_outputs2[1:]
    #
    decoder_inputs = RepeatVector(ahead, name= "repeat1")(encoder_outputs2[0])
    #
    decoder_l1 = LSTM(decoder_n, return_sequences=True, name= "decoder1")(decoder_inputs,initial_state = encoder_states1)
    decoder_l2 = LSTM(decoder_n, return_sequences=True, name= "decoder2")(decoder_l1,initial_state = encoder_states2)
    decoder_outputs2 = TimeDistributed(Dense(out_dim, name = "dense1"),name = "time1")(decoder_l2)
    #
    lstm_model = Model(encoder_inputs,decoder_outputs2)
    #
    return lstm_model

def create_simple_classifier_time(number_class = 1, lag = 500, in_dim = 12):
    
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(lag, in_dim)))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(30, activation='relu'))
    model.add(Dense(number_class, activation='softmax'))
    
    return model

def create_simple_classifier_frequency(number_class = 2, in_dim = 50):

    model= Sequential()
    model.add(Dense(40, input_dim = in_dim, activation = 'relu'))
    model.add(Dense(number_class, activation='softmax'))
    return model

def create_dropout_predict_function(model, dropout):
    """
    Create a keras function to predict with dropout
    model : keras model
    dropout : fraction dropout to apply to all layers
    
    Returns
    predict_with_dropout : keras function for predicting with dropout
    """
    
    # Load the config of the original model
    conf = model.get_config()
    # Add the specified dropout to all layers
    for layer in conf['layers']:
        # Dropout layers
        if layer["class_name"]=="Dropout":
            #print("1")
            layer["config"]["rate"] = dropout
        # Recurrent layers with dropout
        elif "dropout" in layer["config"].keys():
            #print("2")
            #print(layer)
            #print(layer["config"]["dropout"])
            layer["config"]["dropout"] = dropout

    # Create a new model with specified dropout
    if type(model)==Sequential:
        # Sequential
        model_dropout = Sequential.from_config(conf)
    else:
        # Functional
        model_dropout = Model.from_config(conf)
    model_dropout.set_weights(model.get_weights()) 
    
    # Create a function to predict with the dropout on
    predict_with_dropout = K.function(model_dropout.inputs+[K.learning_phase()], model_dropout.outputs)
    
    return predict_with_dropout


def get_stats_from_pred(pred_with_dropout, input_pred, num_iter, ci = 0.8):

    predictions = np.zeros((num_iter, ahead))
    
    for i in range(num_iter):
        
        pred = pred_with_dropout((input_pred ,1.0))
        predictions[i,:] = pred[0].reshape((1,50))
        
    means = predictions.mean(axis=0)
    sds = predictions.std(axis = 0)
    
    lows = np.quantile(predictions, 0.5-ci/2, axis=0)
    uppers = np.quantile(predictions, 0.5+ci/2, axis=0)
    
    return means, sds, lows, uppers


def get_mean_std_ci_from_pred(pred_with_dropout, input_pred, num_iter = 30, ci = 0.8):
    
    predictions = np.zeros((num_iter, ahead,3))

    for i in range(num_iter):
        pred = pred_with_dropout((input_pred ,1.0))
        predictions[i,:] = pred[0].reshape((1,ahead,3))

    means_x = predictions[:,:,0].reshape((-1,100)).mean(axis=0)
    means_y = predictions[:,:,1].reshape((-1,100)).mean(axis=0)
    means_z = predictions[:,:,2].reshape((-1,100)).mean(axis=0)

    sds_x = predictions[:,:,0].reshape((-1,100)).std(axis=0)
    sds_y = predictions[:,:,1].reshape((-1,100)).std(axis=0)
    sds_z = predictions[:,:,2].reshape((-1,100)).std(axis=0)


    lows_x = np.quantile(predictions[:,:,0], 0.5- ci /2, axis=0)
    uppers_x = np.quantile(predictions[:,:,0], 0.5+ ci /2, axis=0)

    lows_y = np.quantile(predictions[:,:,1], 0.5- ci /2, axis=0)
    uppers_y = np.quantile(predictions[:,:,1], 0.5+ ci /2, axis=0)

    lows_z = np.quantile(predictions[:,:,2], 0.5- ci /2, axis=0)
    uppers_z = np.quantile(predictions[:,:,2], 0.5+ ci /2, axis=0)
    
    means = [means_x, means_y, means_z]
    sds = [sds_x, sds_y, sds_z]
    lows = [lows_x, lows_y, lows_z]
    uppers = [uppers_x, uppers_y, uppers_z]

    return means, sds, lows, uppers


    


