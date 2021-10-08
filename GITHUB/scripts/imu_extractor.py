import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import os
import json 
import math
from numpy import array, hstack
from scipy import signal as sg
import samplerate


def find_filenames(path_database):
    
    filenames = []
    
    for file in os.listdir(path_database):
        if file.endswith(".json"):
            filenames.append(file)
            
    return filenames

# extract and preprocess the orientation axes
def extract_orientation_df(data):
    
    data
        
    df = pd.DataFrame(index=None, columns= ['timestamps_orien',
                                              'x_orien',
                                              'y_orien',
                                              'z_orien'])
    xs = []
    ys = []
    zs = []
    ts = []
    
    for v in data:
    
        ts.append(float(v['timestamps']))
        xs.append(float(v['x']))
        ys.append(float(v['y']))
        zs.append(float(v['z']))
        
    df['timestamps_orien'] = ts
    df['x_orien'] = xs
    df['y_orien'] = ys
    df['z_orien'] = zs

    # downsample + normalize                
    df = df.sort_values(['timestamps_orien'], ascending=[True])
    
    subSamplingFactor = 4
    samplingFreq = 6400
    cutFreq = samplingFreq // (2 * subSamplingFactor)
    nyquistFreq = samplingFreq // 2
    subSampledFreq = samplingFreq // subSamplingFactor

    w = sg.firwin(65, cutFreq, fs=samplingFreq)
    
    df['x_orien'] = np.convolve(df['x_orien'], w, mode='same')/360.0
    df['y_orien'] = np.convolve(df['y_orien'], w, mode='same')/360.0
    df['z_orien'] = np.convolve(df['z_orien'], w, mode='same')/360.0
    
    return df
        
# extract and preprocess the gravity axes
def extract_gravity_df(data):
    
    df = pd.DataFrame(index=None, columns= ['timestamps_grav',
                                              'x_grav',
                                              'y_grav',
                                              'z_grav'])
    xs = []
    ys = []
    zs = []
    ts = []
    
    for v in data:
    
        ts.append(float(v['timestamps']))
        xs.append(float(v['x']))
        ys.append(float(v['y']))
        zs.append(float(v['z']))
        
    df['timestamps_grav'] = ts
    df['x_grav'] = xs
    df['y_grav'] = ys
    df['z_grav'] = zs
                        
    df = df.sort_values(['timestamps_grav'], ascending=[True])
    
    # upsampling and normalize
    subSamplingFactor = 4
    samplingFreq = 29.5
    cutFreq = samplingFreq // (2 * subSamplingFactor)
    nyquistFreq = samplingFreq // 2
    subSampledFreq = samplingFreq // subSamplingFactor

    w = sg.firwin(11, cutFreq, fs=samplingFreq)
    
    df['x_grav'] = np.convolve(df['x_grav'], w, mode='same')/1.05
    df['y_grav'] = np.convolve(df['y_grav'], w, mode='same')/1.05
    df['z_grav'] = np.convolve(df['z_grav'], w, mode='same')/1.05
    
    return df

    
#-------------------------------------------------------------------------------------------------------------------------  

# extract and preprocess the gyroscope axes    
def extract_fgyr_df(data, is_car = False):
           
    df = pd.DataFrame(index=None, columns= ['x_gyro_f',
                                            'y_gyro_f',
                                            ' z_gyro_f'])
    xs = []
    ys = []
    zs = []
    
    for v in data:
        
        xs.append(float(v['x']))
        ys.append(float(v['y']))
        zs.append(float(v['z']))
        
    df['x_gyro_f'] = xs
    df['y_gyro_f'] = ys
    df['z_gyro_f'] = zs

    if is_car:

        N = 151
        print("filtering median....")
        #df['x_gyro_f'] = np.convolve(df['x_gyro_f'], np.ones(N)/N, mode='same')
        #df['y_gyro_f'] = np.convolve(df['y_gyro_f'], np.ones(N)/N, mode='same')
        #df['z_gyro_f'] = np.convolve(df['z_gyro_f'], np.ones(N)/N, mode='same')
        df['x_gyro_f'] = sg.medfilt(df['x_gyro_f'], kernel_size=N)
        df['y_gyro_f'] = sg.medfilt(df['y_gyro_f'], kernel_size=N)
        df['z_gyro_f'] = sg.medfilt(df['z_gyro_f'], kernel_size=N)

    #only normalize as already filtered when extracted from superdump
    df['x_gyro_f'] = df['x_gyro_f']/40
    df['y_gyro_f'] = df['y_gyro_f']/40
    df['z_gyro_f'] = df['z_gyro_f']/40
    
    return df

#-------------------------------------------------------------------------------------------------------------------------  
 # extract and preprocess the acceleration axes       
def extract_accl_df(data):
     
    df = pd.DataFrame(index=None, columns= ['timestamps_accl',
                                              'x_accl',
                                              'y_accl',
                                             'z_accl'])
    xs = []
    ys = []
    zs = []
    ts = []
    
    for v in data:
        ts.append(float(v['timestamps']))   
        xs.append(float(v['x']))
        ys.append(float(v['y']))
        zs.append(float(v['z']))
        
        
    df['timestamps_accl'] = ts
    #df['timestamps_accl'] = df['timestamps_accl']/1e6
    df['x_accl'] = xs
    df['y_accl'] = ys
    df['z_accl'] = zs

    #downsample and normalize                  
    df = df.sort_values(['timestamps_accl'], ascending=[True])
    
    subSamplingFactor = 4
    samplingFreq = 200
    cutFreq = samplingFreq // (2 * subSamplingFactor)
    nyquistFreq = samplingFreq // 2
    subSampledFreq = samplingFreq // subSamplingFactor

    w = sg.firwin(11, cutFreq, fs=samplingFreq)
    
    df['x_accl'] = np.convolve(df['x_accl'], w, mode='same')/(16*9.81)
    df['y_accl'] = np.convolve(df['y_accl'], w, mode='same')/(16*9.81)
    df['z_accl'] = np.convolve(df['z_accl'], w, mode='same')/(16*9.81)

    
    return df



def round_decimals_up(number:float, decimals:int=2):
    """
    Returns a value rounded up to a specific number of decimal places.
    """
    if not isinstance(decimals, int):
        raise TypeError("decimal places must be an integer")
    elif decimals < 0:
        raise ValueError("decimal places has to be 0 or more")
    elif decimals == 0:
        return math.ceil(number)

    factor = 10 ** decimals
    return math.ceil(number * factor) / factor
        
    
def extract_all_features(data, recording_freq, downsampling_freq, class_value = None, is_car = False):
    
    
    orien_df = extract_orientation_df(data['orientation[°]'])
    grav_df = extract_gravity_df(data['gravity'])
    accl_df = extract_accl_df(data['accl'])
    fgyr_df = extract_fgyr_df(data['filtered_gyro'], is_car)
    
    #secs = fgyr_df.shape[0]/recording_freq # Number of seconds in signal X
    #resampling_count = int(secs * downsampling_freq)

    resampling_count = accl_df.shape[0] #200 Hz
    converter = 'sinc_best'  # or 'sinc_fastest', ...
    
    resampled_orien_xs = samplerate.resample(np.array(orien_df['x_orien']),  round_decimals_up(resampling_count/orien_df.shape[0],7), converter)
    resampled_orien_ys = samplerate.resample(np.array(orien_df['y_orien']),  round_decimals_up(resampling_count/orien_df.shape[0], 7), converter)
    resampled_orien_zs = samplerate.resample(np.array(orien_df['z_orien']), round_decimals_up(resampling_count/orien_df.shape[0],7), converter)
    
    resampled_grav_xs = samplerate.resample(np.array(grav_df['x_grav']), round_decimals_up(resampling_count/grav_df.shape[0],5), converter)
    resampled_grav_ys = samplerate.resample(np.array(grav_df['y_grav']), round_decimals_up(resampling_count/grav_df.shape[0],5), converter)
    resampled_grav_zs = samplerate.resample(np.array(grav_df['z_grav']), round_decimals_up(resampling_count/grav_df.shape[0],5), converter)
    
    resampled_accl_xs = samplerate.resample(np.array(accl_df['x_accl']),  round_decimals_up(resampling_count/accl_df.shape[0],5), converter)
    resampled_accl_ys = samplerate.resample(np.array(accl_df['y_accl']),  round_decimals_up(resampling_count/accl_df.shape[0],5), converter)
    resampled_accl_zs = samplerate.resample(np.array(accl_df['z_accl']),  round_decimals_up(resampling_count/accl_df.shape[0],5), converter)
    
    resampled_gyros_xsf = samplerate.resample(np.array(fgyr_df['x_gyro_f']),  round_decimals_up(resampling_count/fgyr_df.shape[0],7), converter)
    resampled_gyros_ysf = samplerate.resample(np.array(fgyr_df['y_gyro_f']),  round_decimals_up(resampling_count/fgyr_df.shape[0],7), converter)
    resampled_gyros_zsf = samplerate.resample(np.array(fgyr_df['z_gyro_f']),  round_decimals_up(resampling_count/fgyr_df.shape[0],7), converter)
    
    #print(resampled_orien_xs.shape, resampled_grav_xs.shape, resampled_accl_xs.shape, resampled_gyros_xsf.shape)
    
    resampled_data = {'x_gyro_f': resampled_gyros_xsf,
                  'y_gyro_f': resampled_gyros_ysf,
                  'z_gyro_f': resampled_gyros_zsf,
                  'x_accl': resampled_accl_xs,
                  'y_accl': resampled_accl_ys,
                  'z_accl': resampled_accl_zs,
                  'x_orien': resampled_orien_xs,
                  'y_orien': resampled_orien_ys,
                  'z_orien': resampled_orien_zs,
                  'x_grav': resampled_grav_xs,
                  'y_grav': resampled_grav_ys,
                  'z_grav': resampled_grav_zs,
                  }
    if class_value != None:
        resampled_data['class'] = np.ones(resampling_count, dtype=int) * class_value

    downsampled_df = pd.DataFrame(resampled_data)
    
    return downsampled_df

    
# read features files extracted from the superdump. Forecasting version ( no classes)    
def get_all_dataframes_for_forecasting(path, recording_freq, downsampling_freq):
    
    filenames = find_filenames(path)

    dfs = []
    
    for file in filenames:
         print(file)
         with open(os.path.join(path, file), 'r') as f:
            data = json.load(f)
            if path == "Database/CarDriving_Features":
                df = extract_all_features(data, recording_freq, downsampling_freq, is_car = True)
                dfs.append(df)
            else:
                df = extract_all_features(data, recording_freq, downsampling_freq, is_car =  False)
                dfs.append(df)

        
    return dfs

# read features files extracted from the superdump. Classification version (classes)    
def get_all_dataframes_for_classification(path, recording_freq, downsampling_freq, class_value):

    filenames = find_filenames(path)
    
    dfs = []
    
    for file in filenames:
         print(file)
         with open(os.path.join(path, file), 'r') as f:
            data = json.load(f)
            if path == "Database/CarDriving_Features":
                df = extract_all_features(data, recording_freq, downsampling_freq, class_value, is_car = True)
                dfs.append(df)
            else:
                df = extract_all_features(data, recording_freq, downsampling_freq, class_value, is_car =  False)
                dfs.append(df)
        
    return dfs

def get_freq_magn(sampled_data, sampling_freq, number_freq = 5):
    
    ft = np.fft.rfft(sampled_data)
    #print(len(sampled_data))
    freqs = np.fft.rfftfreq(len(sampled_data), 1.0/sampling_freq)
    mags = abs(ft)
    #plt.plot(freqs, mags)
    inflection = np.diff(np.sign(np.diff(mags)))
    peaks = (inflection < 0).nonzero()[0] + 1
    top_magn = mags[peaks].argsort()[-number_freq:]
    
    top_peaks = peaks[top_magn]
    freq_tops = freqs[top_peaks]
    mag_tops = mags[top_peaks]
    
    if len(freq_tops) < number_freq:
        missing = number_freq - len(freq_tops)
        freq_tops = np.pad(freq_tops, (0, missing), 'constant')
        mag_tops = np.pad(mag_tops, (0, missing), 'constant')
    return np.concatenate((freq_tops, mag_tops), axis=None)

def transform_dataset_into_freq_magn(dataset, sampling_freq, num_freq):
    
    nb_features = dataset.shape[2]
    lag = dataset.shape[1]
    nb_timeframe = dataset.shape[0]
    
    transformed_dataset = np.zeros((nb_timeframe, nb_features * num_freq * 2))
    
    for i in range(nb_timeframe):
        signal = dataset[i]
        freqs_mags = []    
        for f in range(nb_features):
            signal_feature = signal[:,f]
            freqs_mags.append(get_freq_magn(signal_feature, sampling_freq, num_freq))
        transformed_dataset[i] = np.concatenate(freqs_mags, axis = None)
                              
    return transformed_dataset
            
