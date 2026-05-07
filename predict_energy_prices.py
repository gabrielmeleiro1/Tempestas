
"""
Tempestas Production Pipeline

This script implements a complete prediction pipeline that:
1. Takes new weather data inputs
2. Processes through preprocessing steps
3. Runs Stage 1 model (TCN) to predict wind energy production
4. Feeds those predictions to Stage 2 model (XGBoost)
5. Outputs final price predictions

Usage:
    python predict_energy_prices.py \
        --weather_offshore path/to/new_offshore_weather.csv \
        --weather_onshore path/to/new_onshore_weather.csv \
        --historical_prices path/to/historical_prices.csv \
        --output_file path/to/output_predictions.csv
"""

import os
import argparse
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
import xgboost as xgb
import joblib
from pathlib import Path
from datetime import datetime, timedelta
from tcn import TCN

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

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Tempestas Production Pipeline')
    parser.add_argument('--weather_offshore', required=True, help='Path to new offshore weather data CSV')
    parser.add_argument('--weather_onshore', required=True, help='Path to new onshore weather data CSV')
    parser.add_argument('--historical_prices', required=True, help='Path to historical electricity prices CSV')
    parser.add_argument('--historical_energy', required=True, help='Path to historical energy production data CSV')
    parser.add_argument('--output_file', required=True, help='Path to save output predictions')
    parser.add_argument('--tcn_model', default=str(WIND_MODEL_DIR / 'best_tcn_model_21feat_reg.keras'),
                        help='Path to TCN model')
    parser.add_argument('--xgb_model', default=str(PRICE_MODEL_DIR / 'final_xgboost_price_model_tuned.json'),
                        help='Path to XGBoost model')
    parser.add_argument('--scaler_x', default=str(WIND_MODEL_DIR / 'scaler_x_21feat_RobustScaler.joblib'),
                        help='Path to X scaler')
    parser.add_argument('--scaler_y', default=str(WIND_MODEL_DIR / 'scaler_y_21feat_RobustScaler.joblib'),
                        help='Path to Y scaler')
    return parser.parse_args()

def load_and_preprocess_data(weather_offshore_path, weather_onshore_path, historical_prices_path, historical_energy_path):
    """
    Load and preprocess input data files.
    
    Returns:
        tuple: weather_df, price_df, energy_df
    """
    logger.info("Loading input data files...")
    
    # Load weather data
    offshore_weather_df = pd.read_csv(weather_offshore_path)
    onshore_weather_df = pd.read_csv(weather_onshore_path)
    
    # Convert dates to datetime and set as index
    offshore_weather_df['Timestamp'] = pd.to_datetime(offshore_weather_df['date'], utc=True)
    onshore_weather_df['Timestamp'] = pd.to_datetime(onshore_weather_df['date'], utc=True)
    
    offshore_weather_df = offshore_weather_df.set_index('Timestamp').drop(columns=['date'])
    onshore_weather_df = onshore_weather_df.set_index('Timestamp').drop(columns=['date'])
    
    # Add suffix to weather columns for clarity
    offshore_cols = {col: f"{col}_offshore" for col in offshore_weather_df.columns}
    onshore_cols = {col: f"{col}_onshore" for col in onshore_weather_df.columns}
    
    offshore_weather_df = offshore_weather_df.rename(columns=offshore_cols)
    onshore_weather_df = onshore_weather_df.rename(columns=onshore_cols)
    
    # Merge weather dataframes
    weather_df = pd.merge(
        offshore_weather_df, 
        onshore_weather_df, 
        left_index=True, 
        right_index=True, 
        how='inner'
    )
    
    # Load historical prices
    price_df = pd.read_csv(historical_prices_path)
    price_df['Timestamp'] = pd.to_datetime(price_df['Datetime (UTC)'], utc=True)
    price_df = price_df.set_index('Timestamp').drop(columns=['Datetime (UTC)'])
    price_df = price_df[['Price (EUR/MWhe)']].copy()
    
    # Load historical energy for capacity proxy
    energy_df = pd.read_csv(historical_energy_path)
    energy_df['Timestamp'] = pd.to_datetime(energy_df['Timestamp (UTC)'], utc=True)
    energy_df = energy_df.set_index('Timestamp').drop(columns=['Timestamp (UTC)'])
    
    # Align all data on time index and handle any missing values
    common_dates = weather_df.index.intersection(price_df.index)
    weather_df = weather_df.loc[common_dates].sort_index()
    price_df = price_df.loc[common_dates].sort_index()
    
    logger.info(f"Data loaded and preprocessed. Weather shape: {weather_df.shape}, Price shape: {price_df.shape}")
    
    return weather_df, price_df, energy_df

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

def create_feature_list(weather_df):
    """
    Create the list of features needed for the TCN model.
    
    Args:
        weather_df: DataFrame with weather data
        
    Returns:
        list: Feature names to use
    """
    # These should match the features used during training
    feature_cols = list(BASE_TCN_FEATURE_COLUMNS)
    
    # Add lagged energy features if they exist
    if 'Offshore_Norm_Lag1H' in weather_df.columns:
        feature_cols.append('Offshore_Norm_Lag1H')
    if 'Onshore_Norm_Lag1H' in weather_df.columns:
        feature_cols.append('Onshore_Norm_Lag1H')
    
    return feature_cols

def predict_wind_power(weather_df, price_df, energy_df, tcn_model_path, scaler_x_path, scaler_y_path):
    """
    Run Stage 1 model to predict wind energy production.
    
    Args:
        weather_df: DataFrame with preprocessed weather data
        price_df: DataFrame with price data (for alignment)
        energy_df: DataFrame with historical energy data
        tcn_model_path: Path to TCN model file
        scaler_x_path: Path to X scaler
        scaler_y_path: Path to Y scaler
        
    Returns:
        DataFrame: Predictions with timestamps
    """
    logger.info("Running Stage 1: Wind Power Prediction...")
    
    # Load TCN model
    custom_objects = {'TCN': TCN}
    tcn_model = tf.keras.models.load_model(tcn_model_path, custom_objects=custom_objects)
    
    # Load scalers
    scaler_x = joblib.load(scaler_x_path)
    scaler_y = joblib.load(scaler_y_path)
    
    # Get capacity proxy from historical data
    capacity_proxy = calculate_capacity_proxy(energy_df)
    
    # Add lag feature for first prediction if needed
    if 'Offshore_Norm_Lag1H' not in weather_df.columns:
        # Get the most recent normalized energy values
        most_recent_offshore = energy_df['Wind_Offshore_MW'].iloc[-1] / capacity_proxy
        most_recent_onshore = energy_df['Wind_Onshore_MW'].iloc[-1] / capacity_proxy
        
        weather_df['Offshore_Norm_Lag1H'] = most_recent_offshore
        weather_df['Onshore_Norm_Lag1H'] = most_recent_onshore
    
    # Select features for the model
    feature_cols = tcn_feature_columns(LAG_HOURS)
    X_data = weather_df[feature_cols].values
    
    # Create sequences
    if len(X_data) < SEQUENCE_LENGTH:
        logger.error(f"Not enough data points for sequence creation. Need at least {SEQUENCE_LENGTH}, got {len(X_data)}")
        raise ValueError(f"Not enough data points for sequence creation. Need at least {SEQUENCE_LENGTH}, got {len(X_data)}")
    
    # We need to create rolling sequences for each prediction time
    # For example, if SEQUENCE_LENGTH is 12, to predict t, we need [t-11, t-10, ..., t]
    predictions = []
    timestamps = []
    
    for i in range(SEQUENCE_LENGTH - 1, len(X_data)):
        # Extract sequence
        X_sequence = X_data[i - SEQUENCE_LENGTH + 1 : i + 1]
        
        # Scale sequence
        X_sequence_scaled = np.vstack([scaler_x.transform(X_sequence)])
        
        # Reshape for TCN (batch_size, sequence_len, features)
        X_sequence_scaled = X_sequence_scaled.reshape(1, SEQUENCE_LENGTH, len(feature_cols))
        
        # Get prediction (normalized)
        scaled_normalized_pred = tcn_model.predict(X_sequence_scaled, verbose=0)
        
        # Inverse transform to normalized scale
        normalized_pred = scaler_y.inverse_transform(scaled_normalized_pred)[0]
        
        # Convert to MW using capacity proxy
        actual_mw_pred = normalized_pred * capacity_proxy
        
        # Store prediction with timestamp
        timestamp = weather_df.index[i]
        timestamps.append(timestamp)
        predictions.append(actual_mw_pred)
    
    # Create DataFrame with predictions
    pred_array = np.array(predictions)
    pred_df = pd.DataFrame(
        pred_array,
        columns=['Predicted_Offshore_MW', 'Predicted_Onshore_MW'],
        index=timestamps
    )
    
    logger.info(f"Wind power predictions complete. Shape: {pred_df.shape}")
    return pred_df

def predict_electricity_prices(wind_pred_df, price_df, xgb_model_path):
    """
    Run Stage 2 model to predict electricity prices.
    
    Args:
        wind_pred_df: DataFrame with wind power predictions
        price_df: DataFrame with historical price data
        xgb_model_path: Path to XGBoost model file
        
    Returns:
        DataFrame: Price predictions with timestamps
    """
    logger.info("Running Stage 2: Electricity Price Prediction...")
    
    # Load XGBoost model
    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model(xgb_model_path)
    
    # Merge wind predictions with historical prices
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
    
    # Drop rows with NaN values (from lagging)
    merged_df = merged_df.dropna()
    
    if merged_df.empty:
        logger.error("No data available for prediction after creating lag features.")
        raise ValueError("No data available for prediction after creating lag features.")
    
    # Select features for prediction (same as training)
    feature_cols = price_feature_columns(PRICE_LAGS)
    
    X_pred = merged_df[feature_cols]
    
    # Predict electricity prices
    price_predictions = xgb_model.predict(X_pred)
    
    # Create output DataFrame
    output_df = pd.DataFrame({
        'Timestamp': merged_df.index,
        'Predicted_Electricity_Price_EUR_MWh': price_predictions,
        'Predicted_Offshore_MW': merged_df['Predicted_Offshore_MW'],
        'Predicted_Onshore_MW': merged_df['Predicted_Onshore_MW'],
    })
    
    logger.info(f"Electricity price predictions complete. Shape: {output_df.shape}")
    return output_df

def main():
    """Main execution function."""
    args = parse_args()
    
    # Load and preprocess data
    weather_df, price_df, energy_df = load_and_preprocess_data(
        args.weather_offshore,
        args.weather_onshore,
        args.historical_prices,
        args.historical_energy
    )
    
    # Stage 1: Predict wind power
    wind_pred_df = predict_wind_power(
        weather_df, 
        price_df, 
        energy_df,
        args.tcn_model,
        args.scaler_x,
        args.scaler_y
    )
    
    # Stage 2: Predict electricity prices
    price_pred_df = predict_electricity_prices(
        wind_pred_df,
        price_df,
        args.xgb_model
    )
    
    # Save predictions to output file
    price_pred_df.to_csv(args.output_file, index=False)
    logger.info(f"Predictions saved to {args.output_file}")

if __name__ == "__main__":
    main()
