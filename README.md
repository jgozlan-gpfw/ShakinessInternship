# ShakinessInternship

This internship focused on the forecasting of IMU ( gyroscope axes) for potential GoPro applications.

The first step is to extract the necessary features from recordings using the ISG_TOP\ispw\libgpeis\tools\read_eis_dump
Then use the gyro_orientation_imu_extractor.py file, it creates the necessary csv with the gyro, acceleromter, orientation and gravity data that can be then used for postprocessing. When using the gyro_orientation_imu_extractor script, set -F -f as argument to filter the gyro and save the CSV. 

Among the many experiments conducted, the main following notebook were retained:
1. cluster forecasting new multi => create and train a encoder decoder lstm (multi step multi output)
2. cluster classification new => create and train cluster classification neural network models (one in time, one in the frequency domain).
3. pipeline final => the overall pipeline ( preprocessing, classification, perodocity analysis and then forecasting.
4. softmax analysis => try to detect unforescatable signals using the softmax values of the classification model.
5. dropout analysis => try to build confidence interval for predictions using the dropout method at inference time.
6. analysis acf for forecast => try to create a periodicy metric/indicator using the first peak of the autocorrelation function.

The folder scripts contains common functions used accross the different experiments:
1. IMU extractor, contain the signal preprocessing part ( resampling, normalization, and formatting)
2. train test split multi, take the preprocess signals and split them into a train/test datasets.
3. models: contains the encoder/decoder type architecture for forecasting, possible cluster classification models and the dropout model converter for transforming a model into a function with dropout (for confidnce intervals)
4. baseline: contains the few baseline ( mean, period with fourier and acf, fourier extrapolation and others as well as plotting.
5. evaluation2: contains the metrics function (mae, mse, mape and smape) per forecasted timesteps.

Model analysis folder contains a few model trained and used among different scripts.

Few experiments, contains a few experiments for all types of research conducted ( some may be out of date but might contains interesting results)
