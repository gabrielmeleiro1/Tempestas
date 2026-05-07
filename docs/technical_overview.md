## Dataset Description

  Weather Data

  The project uses weather data from Open-Meteo's Historical Weather API. This data includes hourly weather variables for 23 wind
  turbine clusters across the Netherlands (9 offshore, 14 onshore). The variables include:
  - Wind speed at 100m and 10m
  - Wind direction at 100m and 10m
  - Wind gusts at 10m
  - Surface pressure
  - Temperature at 2m
  - Cloud cover (low, mid, high)
  - Relative humidity
  - Rainfall

  The data is fetched using the load_data.py script, which handles API requests with proper error handling, rate limiting, and
  caching.

  Energy Production Data

  Energy production data comes from two sources:
  1. NED.NL Dataportaal - providing hourly offshore and onshore wind energy production
  2. ENTSO-E Transparency Platform - providing actual generation per production type

  The data spans from 2017 to 2025 and is structured as hourly time series of wind energy generation in megawatts (MW), separated
  into offshore and onshore production.

  Electricity Price Data

  Wholesale electricity price data for the Netherlands was obtained from Ember Energy's data repository. This provides hourly
  electricity prices in EUR/MWh from 2015 to 2025.

  Data Processing Pipeline

  Initial Data Collection

  The load_data.py script handles fetching weather data from Open-Meteo for each defined wind turbine location, with proper error
  handling and rate limiting. The energy production data was acquired manually from NED.NL and ENTSO-E and stored in the
  appropriate directories.

  Data Preprocessing

  The preprocessing flow is managed by several scripts:

  1. preprocess_data_script.py - Performs:
    - Time zone conversion and alignment
    - Feature engineering (cyclical encoding of temporal features)
    - Cyclical encoding of wind direction
    - Handling missing values
  2. combine_data.py - Performs:
    - Merging of multiple weather station data for offshore and onshore locations
    - Computing averages across wind turbine clusters
    - Time-based aggregation and alignment
  3. Feature engineering includes:
    - Temporal features (hour, day of week, month, day of year)
    - Cyclical encoding using sine and cosine transformations
    - Rolling capacity proxy calculation
    - Normalization of wind power output
    - Lagged features for target variables

  Wind Power Prediction (Stage 1)

  Model Architecture

  The first stage uses a Temporal Convolutional Network (TCN) based on the keras-tcn implementation. The TCN architecture is
  particularly suited for this task because:

  1. It captures temporal dependencies through dilated convolutions
  2. It allows for parallel processing of sequences (unlike RNNs)
  3. It maintains causality (only using past information)
  4. It handles long-range dependencies efficiently

  The architecture includes:
  - Input layer accepting 12-hour sequences of 21 features
  - TCN layer with 16 filters, kernel size 9, and dilations [1, 2, 4, 8, 16]
  - 4 stacked residual blocks with skip connections
  - Dropout rate of 0.3 for regularization
  - Dense output layer with linear activation
  - Optimization using Adam with learning rate 0.001

  Hyperparameter Tuning

  The azure_hyperparameter_tuning.py script was designed to run on Azure Cloud Compute instances to find optimal hyperparameters
  for the TCN model. It performs a random search over:
  - Sequence length
  - Filter count
  - Kernel size
  - Dropout rate
  - L2 regularization
  - Learning rate
  - Batch size

  The best hyperparameters were stored and used for the final model training.

  Model Training and Inference

  The model was trained on data from 2020-01-01 onward, with validation performed on a chronological split of the dataset. During
  inference (wind_generation_tcn_inference.ipynb), the trained model predicts normalized wind energy production, which is then
  converted back to MW scale using the capacity proxy.

  Electricity Price Prediction (Stage 2)

  Model Architecture

  The second stage uses an XGBoost regression model to predict electricity prices based on:
  1. Predicted wind energy production from Stage 1
  2. Temporal features (hour, day of week, day of year, month, year)
  3. Lagged price values (1h, 2h, 3h, 6h, 12h, 24h)

  XGBoost was chosen for its ability to capture non-linear relationships and feature interactions, as well as its strong
  performance on tabular data.

  Model Training

  The XGBoost model was trained with the following key hyperparameters:
  - 1000 max estimators with early stopping
  - Learning rate: 0.017
  - Max depth: 7
  - Subsample: 0.7
  - L1 regularization (alpha): 0.17
  - L2 regularization (lambda): 3.76e-6

  Feature Importance

  Analysis showed that the most important features for price prediction are:
  1. Predicted offshore wind energy production
  2. Previous hour price (price_lag_1h)
  3. Hour of day
  4. Predicted onshore wind energy production
  5. Day of year

  System Performance

  The two-stage system achieves:
  - MAE of 8.02 EUR/MWh on electricity price prediction
  - RMSE of 12.31 EUR/MWh on electricity price prediction

  This substantially outperforms baseline models:
  - Persistence (previous hour) baseline: MAE 15.18, RMSE 21.53
  - Average baseline: MAE 36.87, RMSE 47.48

  The model effectively captures daily and weekly patterns in electricity prices and their relationship with wind energy
  production. It performs particularly well during normal market conditions but shows higher errors during extreme price events.

  This two-stage approach demonstrates the value of combining physical (weather-based) and statistical models for energy price
  forecasting, creating a more robust prediction system than either approach alone.



##  Tempestas Implementation Guide: End-to-End Process

  This guide provides a step-by-step process for implementing the Tempestas wind energy production and electricity price
  prediction system with your own similar dataset.

  1. Data Collection

  1.1. Weather Data Collection

  - Identify coordinates for your wind farm locations (latitude/longitude)
  - Run load_data.py with your locations to fetch weather variables:
  

  1.2. Energy Production Data

  - Obtain hourly energy production data from your grid operator or energy authority
  - Format it with columns: Timestamp (UTC), Wind_Offshore_MW, Wind_Onshore_MW
  - Save to datasets/raw_energy_production_data/

  1.3. Electricity Price Data

  - Obtain hourly wholesale electricity price data
  - Format with columns: Datetime (UTC), Price (EUR/MWhe)
  - Save to datasets/energy_price/your_region_wholesale_electricity_price_data_hourly.csv

  2. Data Preprocessing

  2.1. Weather Data Preprocessing

  - Run the preprocessing script for weather data:
  python data_processing/preprocess_data_script.py --weather_dir datasets/wind_turbine_clusters_hourly_features --output_dir
  datasets/processed_weather_data
  - This will:
    - Convert timestamps to UTC
    - Apply cyclical encoding to time features (hour, day, month)
    - Apply cyclical encoding to wind direction
    - Handle missing values

  2.2. Energy Data Preprocessing

  - Process the energy production data:
  python data_processing/preprocess_data_script.py --energy_dir datasets/raw_energy_production_data --output_dir
  datasets/processed_energy_data
  - This will:
    - Convert timestamps to UTC
    - Align energy data to hourly intervals
    - Rename columns to standardized format
    - Handle missing values through interpolation

  2.3. Data Combination

  - Combine and average processed weather data:
  python data_processing/combine_data.py
  - This creates:
    - combined_onshore_weather.csv - averaged weather for onshore sites
    - combined_offshore_weather.csv - averaged weather for offshore sites

  2.4. Final Dataset Creation

  - Create the unified dataset:
  python data_processing/stack_energy_data.py
  - This merges:
    - Energy production data
    - Weather data (onshore and offshore)
    - Creates a file in datasets/final_datasets/combined_total_energy_data_2017_2025.csv

  3. Wind Power Prediction Model (Stage 1)

  3.1. Model Setup

  - Configure model parameters in wind_power_prediction/config.py:
    - SEQUENCE_LENGTH = 12 (hours of weather history to use)
    - TCN architecture parameters
    - Define features to use/exclude

  3.2. Hyperparameter Tuning (Optional)

  - For optimal parameters, run hyperparameter tuning script:
  python wind_power_prediction/azure_hyperparameter_tuning.py \
    --energy_data datasets/final_datasets/combined_total_energy_data_2017_2025.csv \
    --onshore_weather datasets/final_datasets/final_averaged_onshore_weather.csv \
    --offshore_weather datasets/final_datasets/final_averaged_offshore_weather.csv \
    --output_dir wind_power_prediction/tuning_results/
  - This performs random search across model configurations
  - Then, update wind_power_prediction/config.py with best parameters

  3.3. Model Training

  - Train the TCN model:
  python wind_power_prediction/main_train.py --start_date 2020-01-01
  - This will:
    - Load and preprocess data as configured
    - Calculate capacity proxy
    - Normalize wind energy values
    - Create sequences for the TCN
    - Train model and save to wind_power_prediction/best_tcn_model_XXfeat_reg.keras
    - Save scalers to wind_power_prediction/scaler_x_XXfeat_RobustScaler.joblib

  3.4. Wind Production Inference

  - Run the inference script:
  python -m jupyter nbconvert --execute --to notebook --inplace energy_price_prediction/wind_generation_tcn_inference.ipynb
  - This generates predictions for wind energy production
  - Output: energy_price_prediction/stage1_predictions_mw_model_trained_from_2020-01-01.csv

  4. Electricity Price Prediction Model (Stage 2)

  4.1. Feature Preparation

  - Open and run energy_price_prediction/xgboost_price_prediction.ipynb
  - This will:
    - Load the predicted wind production from Stage 1
    - Load the electricity price data
    - Merge datasets on timestamp
    - Create time features (hour, day, month, etc.)
    - Create lagged price features (1h, 2h, 3h, 6h, 12h, 24h)

  4.2. XGBoost Model Training

  - Still in the same notebook, configure and train the XGBoost model:
  XGB_PARAMS = {
      'objective': 'reg:squarederror',
      'eval_metric': 'rmse',
      'n_estimators': 1000,
      'learning_rate': 0.017,
      'max_depth': 7,
      'subsample': 0.7,
      'colsample_bytree': 1.0,
      'gamma': 4.49,
      'reg_alpha': 0.17,
      'reg_lambda': 3.76e-6,
      'n_jobs': -1,
      'early_stopping_rounds': 50,
      'random_state': 42
  }
  - Split data chronologically (keep most recent for testing)
  - Train the model on historical data
  - Validate on the test set
  - Save model for production use
  model.save_model("energy_price_prediction/final_xgboost_price_model_tuned.json")

  4.3. Model Evaluation

  - Evaluate model performance:
    - Calculate MAE and RMSE on test set
    - Compare against baseline models (persistence, average)
    - Visualize predictions vs. actual prices
    - Analyze feature importance

  5. Production Deployment

  5.1. Create Prediction Pipeline

  - Implement a production pipeline script that:
    - Takes new weather data inputs
    - Processes through the same preprocessing steps
    - Runs Stage 1 model (TCN) to predict wind energy
    - Feeds those predictions to Stage 2 model (XGBoost)
    - Outputs final price predictions

  5.2. Schedule Regular Predictions

  - Set up a cron job or scheduler to:
    - Fetch latest weather forecasts
    - Run through prediction pipeline
    - Store predictions in database or file system

  5.3. Monitor Model Performance

  - Implement tracking of actual vs. predicted values
  - Set up alerts for significant prediction errors
  - Schedule periodic model retraining as new data becomes available

