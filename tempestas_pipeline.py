#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tempestas Complete Pipeline

This script takes raw weather data files for multiple on/offshore wind farm locations,
performs all necessary preprocessing and prediction steps, and outputs electricity price
predictions in one end-to-end process.

Input:
    - Raw weather CSV files in the current directory:
      - Offshore locations: off_*.csv
      - Onshore locations: on_*.csv
    - Historical electricity prices
    - Historical energy production data for capacity proxy calculation

Output:
    - CSV file with predicted electricity prices

Usage:
    python tempestas_pipeline.py \
        --historical_prices path/to/nl_wholesale_electricity_price_data_hourly.csv \
        --historical_energy path/to/energy_data.csv \
        --output_file predictions.csv \
        [--tcn_model path/to/tcn_model.keras] \
        [--xgb_model path/to/xgb_model.json]
"""

import os
import glob
import argparse
import logging
import time
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
import xgboost as xgb
import joblib
from pathlib import Path
from datetime import datetime, timedelta
from tcn import TCN
from sklearn.preprocessing import RobustScaler

from wind_power_prediction.feature_schema import (
    BASE_TCN_FEATURE_COLUMNS,
    price_feature_columns,
    tcn_feature_columns,
)
from wind_power_prediction.paths import PRICE_MODEL_DIR, WIND_MODEL_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants for the pipeline
SEQUENCE_LENGTH = 12
NUM_FEATURES = 21
NUM_OUTPUTS = 2
SCALER_TYPE = 'RobustScaler'
CAPACITY_PROXY_WINDOW = '365D'
LAG_HOURS = 1
EPSILON = 1e-6
PRICE_LAGS = [1, 2, 3, 6, 12, 24]
LOCAL_TZ = 'Europe/Amsterdam'

# Define the mapping from raw column names to expected model feature names
# Based on observation of the raw data files and what the model expects
FEATURE_MAP = {
    # Offshore mappings
    'wind_speed_100m': 'wind_speed_100m',
    'surface_pressure': 'surface_pressure',
    'temperature_2m': 'temperature_2m',
    'relative_humidity_2m': 'relative_humidity_2m',
    'rain': 'rain',
    'hour_sin': 'hour_sin',
    'hour_cos': 'hour_cos',
    'day_of_week_sin': 'day_of_week_sin',
    'day_of_week_cos': 'day_of_week_cos',
    'day_of_year_sin': 'day_of_year_sin',
    'day_of_year_cos': 'day_of_year_cos',
    'wind_direction_100m_sin': 'wind_direction_100m_sin',
    'wind_direction_100m_cos': 'wind_direction_100m_cos',
}

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Tempestas Complete Pipeline')
    parser.add_argument('--historical_prices', required=True, 
                       help='Path to historical electricity prices CSV (needed for price lags)')
    parser.add_argument('--historical_energy', required=True, 
                       help='Path to historical energy production data CSV (needed for capacity proxy)')
    parser.add_argument('--output_file', default='tempestas_predictions.csv',
                       help='Path to save output predictions')
    parser.add_argument('--tcn_model', default=str(WIND_MODEL_DIR / 'best_tcn_model_21feat_reg.keras'),
                       help='Path to TCN model')
    parser.add_argument('--xgb_model', default=str(PRICE_MODEL_DIR / 'final_xgboost_price_model_tuned.json'),
                       help='Path to XGBoost model')
    parser.add_argument('--scaler_x', default=str(WIND_MODEL_DIR / 'scaler_x_21feat_RobustScaler.joblib'),
                       help='Path to X scaler')
    parser.add_argument('--scaler_y', default=str(WIND_MODEL_DIR / 'scaler_y_21feat_RobustScaler.joblib'),
                       help='Path to Y scaler')
    return parser.parse_args()

def find_weather_files():
    """Find all weather files in the current directory."""
    offshore_files = glob.glob('off_*.csv')
    onshore_files = glob.glob('on_*.csv')
    
    if not offshore_files:
        logger.error("No offshore weather files found (off_*.csv)")
        raise FileNotFoundError("No offshore weather files found (off_*.csv)")
    
    if not onshore_files:
        logger.error("No onshore weather files found (on_*.csv)")
        raise FileNotFoundError("No onshore weather files found (on_*.csv)")
    
    logger.info(f"Found {len(offshore_files)} offshore weather files and {len(onshore_files)} onshore weather files")
    return offshore_files, onshore_files

def process_weather_file(file_path):
    """
    Process a single weather file to add cyclical encoding and time features.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame: Processed weather data
    """
    logger.debug(f"Processing weather file: {file_path}")
    
    try:
        weather_df = pd.read_csv(file_path)
        
        # Convert date column
        weather_df['date'] = pd.to_datetime(weather_df['date'])
        
        # Extract time components
        date_source = weather_df['date'].dt
        weather_df['hour'] = date_source.hour
        weather_df['day_of_week'] = date_source.dayofweek
        weather_df['month'] = date_source.month
        weather_df['day_of_year'] = date_source.dayofyear
        
        # Apply cyclical encoding for time features
        weather_df['hour_sin'] = np.sin(2 * np.pi * weather_df['hour'] / 24.0)
        weather_df['hour_cos'] = np.cos(2 * np.pi * weather_df['hour'] / 24.0)
        weather_df['day_of_week_sin'] = np.sin(2 * np.pi * weather_df['day_of_week'] / 7.0)
        weather_df['day_of_week_cos'] = np.cos(2 * np.pi * weather_df['day_of_week'] / 7.0)
        weather_df['month_sin'] = np.sin(2 * np.pi * (weather_df['month'] - 1) / 12.0)
        weather_df['month_cos'] = np.cos(2 * np.pi * (weather_df['month'] - 1) / 12.0)
        
        # Determine leap year for accurate day_of_year encoding divisor
        is_leap = weather_df['date'].dt.is_leap_year
        days_in_year = np.where(is_leap, 366.0, 365.0)
        weather_df['day_of_year_sin'] = np.sin(2 * np.pi * (weather_df['day_of_year'] - 1) / days_in_year)
        weather_df['day_of_year_cos'] = np.cos(2 * np.pi * (weather_df['day_of_year'] - 1) / days_in_year)
        
        # Apply cyclical encoding for wind direction features
        for col in ['wind_direction_100m', 'wind_direction_10m']:
            if col in weather_df.columns:
                rad = np.deg2rad(weather_df[col])
                weather_df[f'{col}_sin'] = np.sin(rad)
                weather_df[f'{col}_cos'] = np.cos(rad)
                # Drop original direction column as we've encoded it
                weather_df = weather_df.drop(columns=[col])
        
        # Drop original time columns
        cols_to_drop = ['hour', 'day_of_week', 'month', 'day_of_year']
        weather_df = weather_df.drop(columns=cols_to_drop, errors='ignore')
        
        return weather_df
    
    except Exception as e:
        logger.error(f"Failed to process weather file {file_path}: {e}")
        raise

def aggregate_and_average_weather(file_paths, file_type_label):
    """
    Process and combine multiple weather files, averaging by timestamp.
    
    Args:
        file_paths: List of CSV file paths
        file_type_label: Label for logging ('offshore' or 'onshore')
        
    Returns:
        DataFrame: Combined processed weather data
    """
    logger.info(f"Processing and combining {len(file_paths)} {file_type_label} weather files...")
    
    all_dataframes = []
    for filepath in file_paths:
        try:
            # Process the weather file
            df = process_weather_file(filepath)
            
            # Convert 'date' to datetime and set as index
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            all_dataframes.append(df)
            logger.debug(f"Successfully processed: {filepath}")
        
        except Exception as e:
            logger.error(f"Failed to process file {filepath}: {e}")
    
    if not all_dataframes:
        raise ValueError(f"No valid {file_type_label} DataFrames could be processed")
    
    # Concatenate all dataframes
    combined_df = pd.concat(all_dataframes)
    
    # Group by index (date) and calculate mean
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)
        try:
            averaged_df = combined_df.groupby(combined_df.index).mean(numeric_only=True)
        except TypeError:
            # Fallback for older pandas versions
            numeric_cols = combined_df.select_dtypes(include=np.number).columns
            averaged_df = combined_df.groupby(combined_df.index)[numeric_cols].mean()
    
    # Sort by time index
    averaged_df.sort_index(inplace=True)
    
    # Ensure index is timezone-aware (UTC)
    if averaged_df.index.tz is None:
        averaged_df.index = averaged_df.index.tz_localize('UTC')
    else:
        averaged_df.index = averaged_df.index.tz_convert('UTC')
    
    logger.info(f"Successfully created combined {file_type_label} weather data with shape {averaged_df.shape}")
    return averaged_df

def load_historical_data(historical_prices_path, historical_energy_path):
    """
    Load historical price and energy data.
    
    Args:
        historical_prices_path: Path to historical prices CSV
        historical_energy_path: Path to historical energy data CSV
        
    Returns:
        tuple: (price_df, energy_df)
    """
    logger.info("Loading historical data...")
    
    # Check if files exist
    for file_path in [historical_prices_path, historical_energy_path]:
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")
    
    # Load historical prices
    try:
        price_df = pd.read_csv(historical_prices_path)
        price_df['Timestamp'] = pd.to_datetime(price_df['Datetime (UTC)'], utc=True)
        price_df = price_df.set_index('Timestamp')
        price_df = price_df[['Price (EUR/MWhe)']].copy()
        logger.info(f"Loaded historical prices, shape: {price_df.shape}")
    except Exception as e:
        logger.error(f"Failed to load historical prices: {e}")
        raise
    
    # Load historical energy
    try:
        energy_df = pd.read_csv(historical_energy_path)
        energy_df['Timestamp'] = pd.to_datetime(energy_df['Timestamp (UTC)'], utc=True)
        energy_df = energy_df.set_index('Timestamp')
        logger.info(f"Loaded historical energy data, shape: {energy_df.shape}")
    except Exception as e:
        logger.error(f"Failed to load historical energy data: {e}")
        raise
    
    return price_df, energy_df

def calculate_capacity_proxy(energy_df, window=CAPACITY_PROXY_WINDOW, epsilon=EPSILON):
    """
    Calculate capacity proxy from historical energy production.
    
    Args:
        energy_df: DataFrame with energy production
        window: Rolling window size for max calculation
        epsilon: Small value to add for numerical stability
        
    Returns:
        float: Capacity proxy value
    """
    logger.info("Calculating capacity proxy...")
    
    # Create Total_Wind_MW column if it doesn't exist
    if 'Total_Wind_MW' not in energy_df.columns and 'Wind_Offshore_MW' in energy_df.columns and 'Wind_Onshore_MW' in energy_df.columns:
        energy_df['Total_Wind_MW'] = energy_df['Wind_Offshore_MW'] + energy_df['Wind_Onshore_MW']
    
    proxy_base_col = 'Total_Wind_MW' if 'Total_Wind_MW' in energy_df.columns else 'Wind_Offshore_MW'
    
    # Calculate capacity proxy using rolling max
    capacity_proxy = energy_df[proxy_base_col].rolling(window=window, min_periods=1).max() + epsilon
    
    # Use the most recent capacity proxy value
    latest_proxy = float(capacity_proxy.iloc[-1])
    logger.info(f"Calculated capacity proxy: {latest_proxy:.2f} MW")
    
    return latest_proxy

def rename_columns_for_model(df, suffix):
    """
    Rename columns to match what the model expects.
    
    Args:
        df: DataFrame to rename columns for
        suffix: Suffix to add ('offshore' or 'onshore')
        
    Returns:
        DataFrame: DataFrame with renamed columns
    """
    renamed_df = df.copy()
    # Create a mapping for this specific dataframe
    rename_map = {}
    
    # Core weather features
    if 'wind_speed_100m' in renamed_df.columns:
        rename_map['wind_speed_100m'] = f'wind_speed_100m_{suffix}'
    if 'surface_pressure' in renamed_df.columns:
        rename_map['surface_pressure'] = f'surface_pressure_{suffix}'
    if 'temperature_2m' in renamed_df.columns:
        rename_map['temperature_2m'] = f'temperature_2m_{suffix}'
    if 'relative_humidity_2m' in renamed_df.columns:
        rename_map['relative_humidity_2m'] = f'relative_humidity_2m_{suffix}'
    if 'rain' in renamed_df.columns:
        rename_map['rain'] = f'rain_{suffix}'
        
    # Cyclical time features
    if 'hour_sin' in renamed_df.columns:
        rename_map['hour_sin'] = f'hour_sin_{suffix}'
    if 'hour_cos' in renamed_df.columns:
        rename_map['hour_cos'] = f'hour_cos_{suffix}'
    if 'day_of_week_sin' in renamed_df.columns:
        rename_map['day_of_week_sin'] = f'day_of_week_sin_{suffix}'
    if 'day_of_week_cos' in renamed_df.columns:
        rename_map['day_of_week_cos'] = f'day_of_week_cos_{suffix}'
    if 'day_of_year_sin' in renamed_df.columns:
        rename_map['day_of_year_sin'] = f'day_of_year_sin_{suffix}'
    if 'day_of_year_cos' in renamed_df.columns:
        rename_map['day_of_year_cos'] = f'day_of_year_cos_{suffix}'
        
    # Wind direction features
    if 'wind_direction_100m_sin' in renamed_df.columns:
        rename_map['wind_direction_100m_sin'] = f'wind_direction_100m_sin_{suffix}'
    if 'wind_direction_100m_cos' in renamed_df.columns:
        rename_map['wind_direction_100m_cos'] = f'wind_direction_100m_cos_{suffix}'
    
    # Apply the renaming
    renamed_df = renamed_df.rename(columns=rename_map)
    logger.debug(f"Renamed columns for {suffix} data")
    return renamed_df

def prepare_final_dataset(offshore_df, onshore_df):
    """
    Merge offshore and onshore data and prepare final dataset.
    
    Args:
        offshore_df: DataFrame with processed offshore weather
        onshore_df: DataFrame with processed onshore weather
        
    Returns:
        DataFrame: Final dataset ready for prediction
    """
    logger.info("Preparing final dataset for prediction...")
    
    # Rename columns to match what the model expects
    offshore_df_renamed = rename_columns_for_model(offshore_df, 'offshore')
    onshore_df_renamed = rename_columns_for_model(onshore_df, 'onshore')
    
    # Merge the dataframes on the index (timestamp)
    final_df = pd.merge(
        offshore_df_renamed,
        onshore_df_renamed,
        left_index=True,
        right_index=True,
        how='inner'
    )
    
    # Fill any missing values
    final_df = final_df.ffill().bfill()
    
    # Define the feature set needed for the TCN model
    required_features = BASE_TCN_FEATURE_COLUMNS
    
    # Check if we have all required features or can create them
    missing_features = []
    for feature in required_features:
        if feature not in final_df.columns:
            # Try to create this feature by using a reasonable alternative or default value
            if '_sin_' in feature or '_cos_' in feature:
                # For missing trigonometric features, use a neutral value
                if '_sin_' in feature:
                    final_df[feature] = 0.0  # sin(0) = 0
                    logger.warning(f"Created missing feature {feature} with default value 0.0")
                else:
                    final_df[feature] = 1.0  # cos(0) = 1
                    logger.warning(f"Created missing feature {feature} with default value 1.0")
            elif feature.startswith('wind_speed'):
                # Use the other wind speed if one is missing
                if 'offshore' in feature and 'wind_speed_100m_onshore' in final_df.columns:
                    final_df[feature] = final_df['wind_speed_100m_onshore']
                    logger.warning(f"Created missing feature {feature} using onshore values")
                elif 'onshore' in feature and 'wind_speed_100m_offshore' in final_df.columns:
                    final_df[feature] = final_df['wind_speed_100m_offshore']
                    logger.warning(f"Created missing feature {feature} using offshore values")
                else:
                    # If nothing else is available, use a default value
                    final_df[feature] = 10.0  # Average wind speed
                    logger.warning(f"Created missing feature {feature} with default value 10.0")
            elif 'temperature' in feature:
                # Default temperature
                final_df[feature] = 10.0  # 10°C is a reasonable default
                logger.warning(f"Created missing feature {feature} with default value 10.0")
            elif 'humidity' in feature:
                # Default humidity
                final_df[feature] = 70.0  # 70% is a reasonable default
                logger.warning(f"Created missing feature {feature} with default value 70.0")
            elif 'pressure' in feature:
                # Default pressure
                final_df[feature] = 1013.0  # Standard atmospheric pressure
                logger.warning(f"Created missing feature {feature} with default value 1013.0")
            elif 'rain' in feature:
                # Default rain
                final_df[feature] = 0.0  # Default to no rain
                logger.warning(f"Created missing feature {feature} with default value 0.0")
            else:
                # If we can't create a reasonable alternative, add to missing features
                missing_features.append(feature)
    
    # If we still have missing features that we couldn't create, raise an error
    if missing_features:
        logger.error(f"Missing required features that couldn't be created: {missing_features}")
        raise ValueError(f"Missing required features for TCN model: {missing_features}")
    
    logger.info(f"Final dataset prepared with shape: {final_df.shape}")
    return final_df

def create_lagged_features(final_df, energy_df, capacity_proxy):
    """
    Create lagged features for the model.
    
    Args:
        final_df: DataFrame with merged weather data
        energy_df: DataFrame with historical energy data
        capacity_proxy: Calculated capacity proxy value
        
    Returns:
        DataFrame: Dataset with lagged features
    """
    logger.info("Creating lagged features...")
    
    # Get most recent energy values from historical data
    # We need these to create the initial lagged features
    aligned_energy = energy_df[energy_df.index < final_df.index.min()]
    
    if aligned_energy.empty:
        logger.warning("No historical energy data before prediction period. Using default values.")
        most_recent_offshore = 0.3  # Reasonable default based on average capacity factor
        most_recent_onshore = 0.3
    else:
        most_recent_offshore = aligned_energy['Wind_Offshore_MW'].iloc[-1] / capacity_proxy
        most_recent_onshore = aligned_energy['Wind_Onshore_MW'].iloc[-1] / capacity_proxy
    
    # Create normalized target columns for our first lagged features
    final_df['Offshore_Norm'] = np.nan
    final_df['Onshore_Norm'] = np.nan
    
    # Set the first values to create the lag
    first_idx = final_df.index[0]
    final_df.at[first_idx, 'Offshore_Norm'] = most_recent_offshore
    final_df.at[first_idx, 'Onshore_Norm'] = most_recent_onshore
    
    # Create lagged features
    final_df[f'Offshore_Norm_Lag{LAG_HOURS}H'] = final_df['Offshore_Norm'].shift(LAG_HOURS)
    final_df[f'Onshore_Norm_Lag{LAG_HOURS}H'] = final_df['Onshore_Norm'].shift(LAG_HOURS)
    
    # Fill missing lagged values for the first entry
    final_df[f'Offshore_Norm_Lag{LAG_HOURS}H'] = final_df[f'Offshore_Norm_Lag{LAG_HOURS}H'].fillna(most_recent_offshore) 
    final_df[f'Onshore_Norm_Lag{LAG_HOURS}H'] = final_df[f'Onshore_Norm_Lag{LAG_HOURS}H'].fillna(most_recent_onshore)
    
    return final_df

def predict_wind_power(final_df, capacity_proxy, tcn_model_path, scaler_x_path, scaler_y_path):
    """
    Run Stage 1 model to predict wind energy production.
    
    Args:
        final_df: DataFrame with preprocessed features
        capacity_proxy: Capacity proxy value
        tcn_model_path: Path to TCN model
        scaler_x_path: Path to X scaler
        scaler_y_path: Path to Y scaler
        
    Returns:
        DataFrame: Wind power predictions
    """
    logger.info("Running Stage 1: Wind Power Prediction...")
    
    # Check if model files exist
    for file_path in [tcn_model_path, scaler_x_path, scaler_y_path]:
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")
    
    # Load TCN model
    custom_objects = {'TCN': TCN}
    try:
        tcn_model = tf.keras.models.load_model(tcn_model_path, custom_objects=custom_objects)
        logger.info(f"Successfully loaded TCN model from {tcn_model_path}")
    except Exception as e:
        logger.error(f"Failed to load TCN model: {e}")
        raise
    
    # Load scalers
    try:
        scaler_x = joblib.load(scaler_x_path)
        scaler_y = joblib.load(scaler_y_path)
        logger.info(f"Successfully loaded scalers from {scaler_x_path} and {scaler_y_path}")
    except Exception as e:
        logger.error(f"Failed to load scalers: {e}")
        raise
    
    # Define feature columns
    feature_cols = tcn_feature_columns(LAG_HOURS)
    
    # Iterative prediction approach since we need to use our predictions as lags
    wind_predictions = []
    timestamps = []
    
    # Prepare initial data
    current_df = final_df.copy()
    
    # For the number of predictions we want to make
    for i in range(len(final_df) - SEQUENCE_LENGTH + 1):
        # Get current window of data
        if i == 0:
            # First prediction uses the initial lagged values
            current_window = current_df.iloc[:SEQUENCE_LENGTH][feature_cols].values
        else:
            # Update the window by sliding one position
            current_window = current_df.iloc[i:i+SEQUENCE_LENGTH][feature_cols].values
        
        # Scale the input
        current_window_scaled = scaler_x.transform(current_window)
        
        # Reshape for TCN model (batch_size, sequence_length, features)
        X_seq = current_window_scaled.reshape(1, SEQUENCE_LENGTH, len(feature_cols))
        
        # Make prediction
        y_scaled_pred = tcn_model.predict(X_seq, verbose=0)
        
        # Inverse scale to get normalized value
        y_norm_pred = scaler_y.inverse_transform(y_scaled_pred)[0]
        
        # Convert to MW using capacity proxy
        y_mw_pred = y_norm_pred * capacity_proxy
        
        # Store prediction
        timestamp = current_df.index[i + SEQUENCE_LENGTH - 1]
        timestamps.append(timestamp)
        wind_predictions.append(y_mw_pred)
        
        # Update the dataframe with new prediction for next iteration (if not the last one)
        if i < len(final_df) - SEQUENCE_LENGTH:
            next_idx = current_df.index[i + SEQUENCE_LENGTH]
            current_df.at[next_idx, 'Offshore_Norm'] = y_norm_pred[0]
            current_df.at[next_idx, 'Onshore_Norm'] = y_norm_pred[1]
            
            # Update the lag values for the next timestep
            lag_idx = current_df.index[i + SEQUENCE_LENGTH + LAG_HOURS - 1]
            if lag_idx in current_df.index:
                current_df.at[lag_idx, f'Offshore_Norm_Lag{LAG_HOURS}H'] = y_norm_pred[0]
                current_df.at[lag_idx, f'Onshore_Norm_Lag{LAG_HOURS}H'] = y_norm_pred[1]
    
    # Create DataFrame with predictions
    wind_pred_df = pd.DataFrame(
        np.array(wind_predictions),
        columns=['Predicted_Offshore_MW', 'Predicted_Onshore_MW'],
        index=timestamps
    )
    
    logger.info(f"Wind power prediction complete. Generated {len(wind_pred_df)} predictions.")
    return wind_pred_df

def create_price_features(wind_pred_df, price_df):
    """
    Create features for electricity price prediction.
    
    Args:
        wind_pred_df: DataFrame with wind power predictions
        price_df: DataFrame with historical price data
        
    Returns:
        DataFrame: Dataset ready for price prediction
    """
    logger.info("Creating features for electricity price prediction...")
    
    # Merge wind predictions with historical prices
    # Use left join to keep all wind predictions even if we don't have historical prices
    merged_df = pd.merge(
        wind_pred_df,
        price_df,
        left_index=True,
        right_index=True,
        how='left'
    )
    
    # Create time features
    merged_df['hour'] = merged_df.index.hour
    merged_df['dayofweek'] = merged_df.index.dayofweek
    merged_df['dayofyear'] = merged_df.index.dayofyear
    merged_df['month'] = merged_df.index.month
    merged_df['year'] = merged_df.index.year
    merged_df['weekofyear'] = merged_df.index.isocalendar().week.astype(int)
    
    # Create lagged price features
    price_col = 'Price (EUR/MWhe)'
    
    for lag in PRICE_LAGS:
        merged_df[f'price_lag_{lag}h'] = merged_df[price_col].shift(lag)
    
    # If we don't have all the lagged values (at the beginning of the series),
    # we'll need to get them from historical data
    missing_lags = merged_df[f'price_lag_{max(PRICE_LAGS)}h'].isna().any()
    
    if missing_lags:
        logger.warning("Missing lagged price values. Will attempt to fill from historical data.")
        
        # Find the earliest timestamp with missing lag values
        min_timestamp = merged_df.index.min()
        max_lag_hours = max(PRICE_LAGS)
        
        # Get historical prices covering the lag period
        lag_start = min_timestamp - pd.Timedelta(hours=max_lag_hours)
        historical_prices = price_df.loc[lag_start:min_timestamp].copy()
        
        if not historical_prices.empty:
            # For each lag, fill forward the historical values
            for lag in PRICE_LAGS:
                lag_values = {}
                
                for i, idx in enumerate(merged_df.index[:lag]):
                    # Calculate the historical timestamp we need
                    hist_idx = idx - pd.Timedelta(hours=lag)
                    
                    if hist_idx in historical_prices.index:
                        lag_values[idx] = historical_prices.loc[hist_idx, price_col]
                
                # Update the dataframe with these values
                for idx, value in lag_values.items():
                    merged_df.at[idx, f'price_lag_{lag}h'] = value
    
    # Drop rows that still have NaN values
    rows_before = len(merged_df)
    merged_df = merged_df.dropna()
    rows_after = len(merged_df)
    
    if rows_before > rows_after:
        logger.warning(f"Dropped {rows_before - rows_after} rows due to missing price lag values.")
    
    logger.info(f"Price features created. Final shape: {merged_df.shape}")
    return merged_df

def predict_electricity_prices(price_features_df, xgb_model_path):
    """
    Run Stage 2 model to predict electricity prices.
    
    Args:
        price_features_df: DataFrame with features for price prediction
        xgb_model_path: Path to XGBoost model
        
    Returns:
        DataFrame: Final price predictions
    """
    logger.info("Running Stage 2: Electricity Price Prediction...")
    
    # Check if model file exists
    if not os.path.exists(xgb_model_path):
        logger.error(f"XGBoost model file not found: {xgb_model_path}")
        raise FileNotFoundError(f"XGBoost model file not found: {xgb_model_path}")
    
    # Load XGBoost model
    try:
        xgb_model = xgb.XGBRegressor()
        xgb_model.load_model(xgb_model_path)
        logger.info(f"Successfully loaded XGBoost model from {xgb_model_path}")
    except Exception as e:
        logger.error(f"Failed to load XGBoost model: {e}")
        raise
    
    # Define feature columns (same as used in training)
    feature_cols = price_feature_columns(PRICE_LAGS)
    
    # Check if we have all required features
    missing_features = [col for col in feature_cols if col not in price_features_df.columns]
    if missing_features:
        logger.error(f"Missing required features for XGBoost model: {missing_features}")
        raise ValueError(f"Missing required features for XGBoost model: {missing_features}")
    
    # Extract features
    X_pred = price_features_df[feature_cols]
    
    # Predict electricity prices
    price_predictions = xgb_model.predict(X_pred)
    
    # Create output DataFrame
    output_df = pd.DataFrame({
        'Timestamp': price_features_df.index,
        'Predicted_Electricity_Price_EUR_MWh': price_predictions,
        'Predicted_Offshore_MW': price_features_df['Predicted_Offshore_MW'],
        'Predicted_Onshore_MW': price_features_df['Predicted_Onshore_MW']
    })
    
    # Add actual price if available (for validation)
    if 'Price (EUR/MWhe)' in price_features_df.columns:
        output_df['Actual_Price_EUR_MWh'] = price_features_df['Price (EUR/MWhe)']
    
    logger.info(f"Electricity price prediction complete. Generated {len(output_df)} predictions.")
    return output_df

def main():
    """Main execution function."""
    start_time = time.time()

    # Parse arguments
    args = parse_args()
    logger.info("Starting Tempestas pipeline...")
    
    try:
        # Step 1: Find all weather files
        offshore_files, onshore_files = find_weather_files()
        
        # Step 2: Process and combine offshore files
        offshore_df = aggregate_and_average_weather(offshore_files, 'offshore')
        
        # Step 3: Process and combine onshore files
        onshore_df = aggregate_and_average_weather(onshore_files, 'onshore')
        
        # Step 4: Load historical data
        price_df, energy_df = load_historical_data(args.historical_prices, args.historical_energy)
        
        # Step 5: Calculate capacity proxy
        capacity_proxy = calculate_capacity_proxy(energy_df)
        
        # Step 6: Prepare final dataset
        final_df = prepare_final_dataset(offshore_df, onshore_df)
        
        # Step 7: Create lagged features
        final_df = create_lagged_features(final_df, energy_df, capacity_proxy)
        
        # Step 8: Predict wind power (Stage 1)
        wind_pred_df = predict_wind_power(
            final_df, 
            capacity_proxy,
            args.tcn_model,
            args.scaler_x,
            args.scaler_y
        )
        
        # Step 9: Create price features
        price_features_df = create_price_features(wind_pred_df, price_df)
        
        # Step 10: Predict electricity prices (Stage 2)
        price_pred_df = predict_electricity_prices(price_features_df, args.xgb_model)
        
        # Step 11: Save predictions
        price_pred_df.to_csv(args.output_file, index=False)
        logger.info(f"Predictions saved to {args.output_file}")
        
        # Performance stats
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"Pipeline completed successfully in {elapsed_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
