import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import random

# metrics

def symmetric_mean_absolute_percentage_error(A, F):
    tmp = 2 * np.abs(F - A) / (np.abs(A) + np.abs(F))
    len_ = np.count_nonzero(~np.isnan(tmp))
    if len_ == 0 and np.nansum(tmp) == 0: # Deals with a special case
        return 100
    return 100 / len_ * np.nansum(tmp)

def mean_absolute_percentage_error_new(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def compute_metrics(true,pred):
    
    mse = mean_squared_error(true, pred)
    mae = mean_absolute_error(true, pred)
    mape = mean_absolute_percentage_error_new(true, pred)
    smape = symmetric_mean_absolute_percentage_error(true, pred)
    return mse, mae, mape, smape

def get_metrics_for_forecast(true, forecast):
    
    score_mse = []
    score_mae = []
    score_mape = []
    score_smape = []
    
    for timestep in range(forecast.shape[1]):
        
        true_at_ts = true[:, timestep]
        forecast_at_ts = forecast[:, timestep]
        mse, mae, mape, smape  = compute_metrics(true_at_ts, forecast_at_ts)
        score_mse.append(mse)
        score_mae.append(mae)
        score_mape.append(mape)
        score_smape.append(smape)
    
    return score_mse, score_mae, score_mape, score_smape

def plot_metric(metric_name, list_keys, list_metric, forecasts_color_dict):
    
    plt.figure()
    
    for i in range(len(list_keys)):
        
        forecast_name = list_keys[i]
        plt.plot(range(len(list_metric[i])), list_metric[i], label = forecast_name, color = forecasts_color_dict[forecast_name])
  
    plt.legend(loc = "upper left")
    plt.xlabel("timesteps")
    plt.ylabel(metric_name)
        

# compute metrics per timesteps and plot them
def plot_metrics(true, forecasts_dict, forecasts_color_dict):
    
    list_mse = []
    list_mae = []
    list_mape = []
    list_smape = []
    list_keys = []
    
    
    for key, forecast in forecasts_dict.items():
        
        mse, mae, mape, smapes = get_metrics_for_forecast(true, forecast)
        list_mse.append(mse)
        list_mae.append(mae)
        list_mape.append(mape)
        list_smape.append(smapes)
        list_keys.append(key)
    
    plot_metric("mse", list_keys, list_mse, forecasts_color_dict)
    plot_metric("mae", list_keys, list_mae, forecasts_color_dict)
    plot_metric("mape", list_keys, list_mape,forecasts_color_dict)
    plot_metric("smape", list_keys, list_smape, forecasts_color_dict)

def plot_metrics_multi(list_y_true, list_forecasts_dict, forecasts_color_dict):

    # [x_trye y_true, z_true] , [[forecast, acf], forecast, acf]
    list_forecast_names = list_forecasts_dict[0].keys()

    list_scores_for_axes  = []

    for axe in range(len(list_y_true)):

        list_mse = []
        list_mae = []
        list_mape = []
        list_smape = []
        list_names = []

        for name in list_forecast_names:

            mse, mae, mape, smapes = get_metrics_for_forecast(list_y_true[axe], list_forecasts_dict[axe][name])
            list_mse.append(mse)
            list_mae.append(mae)
            list_mape.append(mape)
            list_smape.append(smapes)
            list_names.append(name)

        
        list_scores_for_axes.append( {"mse": list_mse, "mae": list_mae,"mape": list_mape, "smape": list_smape, "name": list_names})
    
    for metric in ["mse", "mae", "mape","smape"]:
        
        n_axes = len(list_y_true)
        if n_axes > 1:

            fig, axs = plt.subplots(nrows= 1, ncols=len(list_y_true), figsize=(18,4))

            for j, ax in enumerate(axs.flatten()):
                for k,name in enumerate(list_scores_for_axes[j]["name"]):
                    ax.plot(range(list_y_true[j].shape[1]), list_scores_for_axes[j][metric][k] ,label = name, color = forecasts_color_dict[name])

            ax.legend(loc = "upper left")
            ax.set_xlabel("timesteps")
            ax.set_ylabel(metric)
            
        else:
            plt.figure()
            for k,name in enumerate(list_scores_for_axes[0]["name"]):
                plt.plot(range(list_y_true[0].shape[1]), list_scores_for_axes[0][metric][k] ,label = name, color = forecasts_color_dict[name])

            plt.legend(loc = "upper left")
            plt.xlabel("timesteps")
            plt.ylabel(metric)
            
        plt.plot()

def plot_forecasts(X_true, y_true, forecasts, dict_forecasts_color, lag, ahead, target_index, k):
    
    random_indexes = random.sample(range(X_true.shape[0]), k)
    
    for i in random_indexes:
        
        plt.figure()
        
        X_input = X_true[i][:,target_index].reshape((lag,-1))[-ahead:]
        t = [len(X_input) + j for j in range(len(y_true[i]))]

        plt.plot(range(len(X_input)), X_input, label = 'x_gyro input', color = "green")
        plt.plot(t, y_true[i], label = 'x_gyro_true', color = "green")
        
        for name_forecast, forecast in forecasts.items():
            plt.plot(t, forecast[i], label = name_forecast, color = dict_forecasts_color[name_forecast])

        plt.legend(loc="upper left")
        
    plt.show()


