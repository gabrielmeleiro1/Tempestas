#!/usr/bin/env python
# coding: utf-8

"""
Hyperparameter tuning script for the TCN model predicting normalized wind energy generation.

Designed for execution within Azure ML jobs. It expects paths to input data files
(energy, onshore weather, offshore weather) and an output directory to be provided
as command-line arguments. Performs random search over a defined hyperparameter
space, training and evaluating a TCN model for each combination.

Preprocessing steps (merging, feature engineering, etc.) are included.
Results are saved to a CSV file in the output directory.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib 
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tcn import TCN 
import time
import os
import traceback
import random
from tqdm import tqdm 
import argparse # For command-line arguments


try:
    from tqdm.keras import TqdmCallback
    TQDMCALLBACK_AVAILABLE = True
except ImportError:
    TQDMCALLBACK_AVAILABLE = False

# Constants 
# File paths are now provided via argparse, removing old constants

# Feature selection based on previous analysis 
COLS_TO_REMOVE_FROM_FEATURES = [
    'hour_sin_onshore', 'hour_cos_onshore', 'day_of_week_sin_onshore', 'day_of_week_cos_onshore',
    'day_of_year_sin_onshore', 'day_of_year_cos_onshore', 'month_sin_offshore', 'month_cos_offshore',
    'month_sin_onshore', 'month_cos_onshore', 'wind_direction_10m_sin_offshore', 'wind_direction_10m_cos_offshore',
    'wind_direction_10m_sin_onshore', 'wind_direction_10m_cos_onshore', 'wind_speed_10m_offshore',
    'wind_gusts_10m_offshore', 'wind_speed_10m_onshore', 'wind_gusts_10m_onshore', 'surface_pressure_onshore',
    'cloud_cover_low_offshore', 'cloud_cover_mid_offshore', 'cloud_cover_high_offshore', 'cloud_cover_low_onshore',
    'cloud_cover_mid_onshore', 'cloud_cover_high_onshore'
]
TARGET_COLS = ['Offshore_Norm', 'Onshore_Norm']
COLS_TO_EXCLUDE_FROM_FEATURES_BASE = ['Wind_Offshore_MW', 'Wind_Onshore_MW', 'Total_Wind_MW', 'capacity_proxy']


#  Helper: Sequence Creation 
def create_sequences(X_data, y_data, sequence_length):
    """Creates sequences for time series forecasting."""
    X_seq_list, y_seq_list = [], []
    for i in range(len(X_data) - sequence_length):
        X_seq_list.append(X_data[i : i + sequence_length])
        # Target is the value at the end of the sequence window
        y_seq_list.append(y_data[i + sequence_length - 1])
    if not X_seq_list or not y_seq_list:
        return None, None
    return np.array(X_seq_list), np.array(y_seq_list)

#  Helper: Model Definition 
def create_tcn_model(input_shape, num_outputs, nb_filters, kernel_size, dilations, nb_stacks,
                     padding, use_skip_connections, return_sequences, dropout_rate, activation,
                     use_l2_reg, l2_factor, lr, model_name='tcn_model'):
    """Defines and compiles the TCN model."""
    input_layer = Input(shape=input_shape, name="Input_Layer")
    tcn_layer = TCN(nb_filters=nb_filters, kernel_size=kernel_size, nb_stacks=nb_stacks,
                      dilations=dilations, padding=padding, use_skip_connections=use_skip_connections,
                      dropout_rate=dropout_rate, return_sequences=return_sequences, activation=activation,
                      kernel_initializer='he_normal', use_batch_norm=True, # Keeping BN as default
                      use_layer_norm=False, name=model_name)(input_layer)

    output_regularizer = l2(l2_factor) if use_l2_reg else None
    # Only apply Dense if return_sequences=False (TCN outputs shape [batch, features])
    if not return_sequences:
         x = Dense(num_outputs, name="Dense_Output_Regressor", kernel_regularizer=output_regularizer)(tcn_layer)
         output_layer = Activation('linear', name="Linear_Output")(x)
    else:
         raise ValueError("return_sequences=True is not handled correctly by the Dense layer structure. Set return_sequences=False.")


    loss_func = 'mean_squared_error'
    metrics = ['mae'] # Focus on MAE for validation loss

    model = Model(input_layer, output_layer, name=f"{model_name}_Compiled")
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss=loss_func, metrics=metrics)
    return model

# Main Experiment Function 
def run_experiment(
    # --- Data/Preprocessing Params (Paths passed in) ---
    energy_data_path: str,
    onshore_weather_path: str,
    offshore_weather_path: str,
    output_dir: str, # Expect output directory path
    start_date='2020-01-01',
    lag_hours=6,
    scaler_type='RobustScaler',
    validation_split_frac=0.2,
    #  TCN Model Params 
    sequence_length=24,
    tcn_num_filters=32,
    tcn_kernel_size=6,
    tcn_dilations=(1, 2, 4, 8),
    tcn_num_stacks=1,
    padding='causal',
    use_skip_connections=True,
    return_sequences=False, 
    tcn_dropout_rate=0.1,
    activation='relu',
    use_l2_reg=False,
    l2_factor=0.01,
    #  Training Params 
    learning_rate=0.001,
    epochs=50,
    batch_size=64,
    use_lr_scheduler=True,
    #  Control Params 
    verbose=1 # Default verbose level
    ):
    """
    Runs a single training and evaluation experiment with given hyperparameters.
    Reads data from provided paths and saves results to output_dir.

    Returns:
        tuple: (validation_mae_actual, validation_rmse_actual) on the original data scale.
               Returns (np.inf, np.inf) on failure.
    """
    if verbose > 0:
        print(f"\n--- Running Experiment ---")
        # Shortened param print for less clutter during tuning
        print(f"Params: seq={sequence_length}, lag={lag_hours}, filt={tcn_num_filters}, kern={tcn_kernel_size}, drop={tcn_dropout_rate:.2f}, l2={l2_factor if use_l2_reg else 'N/A'}, lr={learning_rate:.5f}")
    start_time = time.time()

    try:
        # Output directory setup
        run_output_dir = Path(output_dir)
        #run_output_dir.mkdir(parents=True, exist_ok=True) # Ensure output dir exists

        # 1. Load Data using argument paths 
        try:
            energy_df = pd.read_csv(energy_data_path)
            onshore_weather_df = pd.read_csv(onshore_weather_path)
            offshore_weather_df = pd.read_csv(offshore_weather_path)
            if verbose > 0: print(f"Loaded data from: {energy_data_path}, {onshore_weather_path}, {offshore_weather_path}")
        except FileNotFoundError as e:
            print(f"ERROR: Data file not found: {e}. Check input paths.")
            return np.inf, np.inf
        except Exception as e:
             print(f"ERROR loading data files: {e}")
             return np.inf, np.inf

        #  2. Preprocessing and Merging 
        energy_df['Timestamp'] = pd.to_datetime(energy_df['Timestamp (UTC)'], utc=True)
        offshore_weather_df['Timestamp'] = pd.to_datetime(offshore_weather_df['date'], utc=True)
        onshore_weather_df['Timestamp'] = pd.to_datetime(onshore_weather_df['date'], utc=True)

        df_targets = energy_df.set_index('Timestamp').drop(columns=['Timestamp (UTC)'])
        df_offshore_weather = offshore_weather_df.set_index('Timestamp').drop(columns=['date'])
        df_onshore_weather = onshore_weather_df.set_index('Timestamp').drop(columns=['date'])

        offshore_cols = {col: f"{col}_offshore" for col in df_offshore_weather.columns}
        onshore_cols = {col: f"{col}_onshore" for col in df_onshore_weather.columns}
        df_offshore_weather = df_offshore_weather.rename(columns=offshore_cols)
        df_onshore_weather = df_onshore_weather.rename(columns=onshore_cols)

        merged_df = pd.merge(df_targets, df_offshore_weather, left_index=True, right_index=True, how='inner')
        final_df = pd.merge(merged_df, df_onshore_weather, left_index=True, right_index=True, how='inner')
        final_df = final_df.sort_index()
        final_df = final_df.ffill().bfill() # Fill any gaps

        # Filter by date *before* calculating rolling/lagged features
        final_df = final_df[final_df.index >= start_date]
        if final_df.empty: raise ValueError(f"No data remaining after filtering for start date {start_date}")

        # Capacity Proxy (Using fixed window for simplicity)
        window = '365D'
        proxy_base_col = 'Wind_Offshore_MW'
        if 'Wind_Offshore_MW' in final_df.columns and 'Wind_Onshore_MW' in final_df.columns:
            final_df['Total_Wind_MW'] = final_df['Wind_Offshore_MW'] + final_df['Wind_Onshore_MW']
            proxy_base_col = 'Total_Wind_MW'
        else:
             if verbose > 0: print(f"Warning: Could not find both 'Wind_Offshore_MW' and 'Wind_Onshore_MW'. Using '{proxy_base_col}' for proxy.")

        # Ensure proxy base column exists
        if proxy_base_col not in final_df.columns:
             raise ValueError(f"Base column for capacity proxy '{proxy_base_col}' not found in data.")

        final_df['capacity_proxy'] = final_df[proxy_base_col].rolling(window=window, min_periods=1).max().ffill().bfill() + 1e-6 # Add epsilon for stability
        final_df['Offshore_Norm'] = (final_df['Wind_Offshore_MW'] / final_df['capacity_proxy']).clip(0, 1.1) # Allow slightly > 1
        final_df['Onshore_Norm'] = (final_df['Wind_Onshore_MW'] / final_df['capacity_proxy']).clip(0, 1.1)

        # Lagged Target Features (using normalized values)
        lag_col_offshore = f'Offshore_Norm_Lag{lag_hours}H'
        lag_col_onshore = f'Onshore_Norm_Lag{lag_hours}H'
        final_df[lag_col_offshore] = final_df['Offshore_Norm'].shift(lag_hours)
        final_df[lag_col_onshore] = final_df['Onshore_Norm'].shift(lag_hours)

        #  3. Separate Features/Targets 
        cols_to_exclude = COLS_TO_EXCLUDE_FROM_FEATURES_BASE + TARGET_COLS
        all_feature_cols = [col for col in final_df.columns if col not in cols_to_exclude]
        X_initial = final_df[all_feature_cols]
        y = final_df[TARGET_COLS]

        #  4. Feature Selection 
        selected_feature_cols = [col for col in X_initial.columns if col not in COLS_TO_REMOVE_FROM_FEATURES]
        X = X_initial[selected_feature_cols].copy()
        NUM_FEATURES = X.shape[1]
        NUM_OUTPUTS = y.shape[1]
        if NUM_FEATURES == 0: raise ValueError("No features selected after removal.")

        #  5. Train/Test Split (Chronological) 
        split_index = int(len(X) * (1 - validation_split_frac))
        if split_index < sequence_length + lag_hours: # Ensure enough data for sequences and lag
             raise ValueError(f"Train split too small ({split_index} samples) for sequence length ({sequence_length}) and lag ({lag_hours}). Reduce validation_split_frac or require more data.")

        X_train_df, y_train_df = X[:split_index], y[:split_index]
        X_test_df, y_test_df = X[split_index:], y[split_index:]

        #  6. Handle NaNs introduced by Lagging & Scaling 
        # Drop NaNs *after* split to avoid data leakage from test set Nan handling into train set
        lag_cols_in_train = [col for col in X_train_df.columns if f'_Lag{lag_hours}H' in col]
        if lag_cols_in_train and X_train_df[lag_cols_in_train].isnull().any().any():
            valid_index_train = X_train_df.dropna(subset=lag_cols_in_train).index
            X_train_df = X_train_df.loc[valid_index_train]
            y_train_df = y_train_df.loc[valid_index_train]
            if X_train_df.empty: raise ValueError("Training data empty after dropping lag NaNs.")

        # Also drop NaNs from test set that would affect sequences later
        lag_cols_in_test = [col for col in X_test_df.columns if f'_Lag{lag_hours}H' in col]
        if lag_cols_in_test and X_test_df[lag_cols_in_test].isnull().any().any():
             valid_index_test = X_test_df.dropna(subset=lag_cols_in_test).index
             X_test_df = X_test_df.loc[valid_index_test]
             y_test_df = y_test_df.loc[valid_index_test]
             if X_test_df.empty: raise ValueError("Test data empty after dropping lag NaNs.")

        # Scaling
        if scaler_type == 'RobustScaler': scaler_x, scaler_y = RobustScaler(), RobustScaler()
        else: scaler_x, scaler_y = StandardScaler(), StandardScaler() # Default to StandardScaler

        X_train_scaled = scaler_x.fit_transform(X_train_df)
        y_train_scaled = scaler_y.fit_transform(y_train_df) # Fit scaler_y only on train targets
        X_test_scaled = scaler_x.transform(X_test_df)
        y_test_scaled = scaler_y.transform(y_test_df)


        #  7. Create Sequences 
        X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, sequence_length)
        X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, sequence_length)

        if X_train_seq is None or X_test_seq is None or X_train_seq.shape[0] == 0 or X_test_seq.shape[0] == 0:
            raise ValueError(f"Sequence creation failed or resulted in empty arrays. Train shape: {X_train_seq.shape if X_train_seq is not None else 'None'}, Test shape: {X_test_seq.shape if X_test_seq is not None else 'None'}. Check sequence_length ({sequence_length}) vs data length after NaN drop.")

        #  8. TCN Model Definition 
        tf.keras.backend.clear_session() # Clear previous models from memory
        model = create_tcn_model(
            input_shape=(sequence_length, NUM_FEATURES),
            num_outputs=NUM_OUTPUTS,
            nb_filters=tcn_num_filters,
            kernel_size=tcn_kernel_size,
            dilations=tcn_dilations,
            nb_stacks=tcn_num_stacks,
            padding=padding,
            use_skip_connections=use_skip_connections,
            return_sequences=return_sequences, # MUST BE FALSE
            dropout_rate=tcn_dropout_rate,
            activation=activation,
            use_l2_reg=use_l2_reg,
            l2_factor=l2_factor,
            lr=learning_rate
        )

        #  9. Model Training 
        early_stopping = EarlyStopping(monitor='val_mae', patience=15, restore_best_weights=True, verbose=0, mode='min')
        callbacks_list = [early_stopping]
        if use_lr_scheduler:
            reduce_lr = ReduceLROnPlateau(monitor='val_mae', factor=0.2, patience=5, min_lr=1e-6, verbose=0, mode='min')
            callbacks_list.append(reduce_lr)

        keras_verbose_level = 0
        if verbose == 1:
            if TQDMCALLBACK_AVAILABLE:
                callbacks_list.append(TqdmCallback(verbose=1))
                keras_verbose_level = 0
            else:
                keras_verbose_level = 2
                if verbose > 0: print("TQDMCallback not found, using Keras default progress per epoch.")
        elif verbose >= 2:
             keras_verbose_level = verbose

        history = model.fit(
            X_train_seq, y_train_seq,
            epochs=epochs, batch_size=batch_size,
            validation_data=(X_test_seq, y_test_seq),
            callbacks=callbacks_list,
            verbose=keras_verbose_level
        )

        #  10. Evaluation 
        scaled_evaluation = model.evaluate(X_test_seq, y_test_seq, batch_size=batch_size * 2, verbose=0)
        scaled_val_mae = scaled_evaluation[1] # MAE

        # Inverse transform to get actual scale MAE/RMSE
        scaled_normalized_predictions = model.predict(X_test_seq, batch_size=batch_size * 2, verbose=0)
        normalized_predictions = scaler_y.inverse_transform(scaled_normalized_predictions)

        # Align capacity proxy with the test sequences
        test_target_indices = y_test_df.index[sequence_length:]
        if len(test_target_indices) != len(y_test_seq):
             if verbose > 0: print(f"Warning: Length mismatch between derived test target indices ({len(test_target_indices)}) and y_test_seq ({len(y_test_seq)}). Adjusting alignment.")
             min_len_eval = min(len(normalized_predictions), len(y_test_seq))
             test_indices_for_proxy = y_test_df.index[-min_len_eval:]
        else:
             min_len_eval = len(y_test_seq)
             test_indices_for_proxy = test_target_indices

        # Extract capacity proxy using the aligned indices from the processed dataframe
        capacity_proxy_test_aligned = final_df.loc[test_indices_for_proxy, 'capacity_proxy'].values.reshape(-1, 1)

        # Ensure all arrays for final metric calculation have the same length
        normalized_predictions = normalized_predictions[:min_len_eval]
        y_test_seq_aligned = y_test_seq[:min_len_eval]
        capacity_proxy_test_aligned = capacity_proxy_test_aligned[:min_len_eval]

        # Inverse transform the *actual* scaled test targets
        normalized_y_test = scaler_y.inverse_transform(y_test_seq_aligned)

        # Convert normalized predictions and targets back to actual MW scale
        actual_scale_predictions = normalized_predictions * capacity_proxy_test_aligned
        actual_scale_y_test = normalized_y_test * capacity_proxy_test_aligned

        # Calculate final metrics on actual scale
        val_mae_actual = mean_absolute_error(actual_scale_y_test, actual_scale_predictions)
        val_rmse_actual = np.sqrt(mean_squared_error(actual_scale_y_test, actual_scale_predictions))

        end_time = time.time()
        if verbose > 0:
            print(f"Finished Experiment. Val MAE (scaled): {scaled_val_mae:.4f}, Val MAE (actual): {val_mae_actual:.4f}, Val RMSE (actual): {val_rmse_actual:.4f}. Time: {end_time - start_time:.2f}s")


        return val_mae_actual, val_rmse_actual

    except Exception as e:
        print(f"!!! Experiment Failed: {e}")
        traceback.print_exc()
        return np.inf, np.inf


#  Main Execution Block 
if __name__ == "__main__":

    # Setup Argument Parser 
    # Defines command-line arguments expected by the script.
    # In Azure ML, these are populated using ${{inputs.*}} and ${{outputs.*}} syntax.
    parser = argparse.ArgumentParser(description="Run TCN Hyperparameter Tuning for Wind Power Forecasting in Azure ML")
    parser.add_argument('--energy_data', type=str, required=True, help='Path to the input combined energy data CSV file (provided by Azure ML).')
    parser.add_argument('--onshore_weather', type=str, required=True, help='Path to the input final averaged onshore weather CSV file (provided by Azure ML).')
    parser.add_argument('--offshore_weather', type=str, required=True, help='Path to the input final averaged offshore weather CSV file (provided by Azure ML).')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory path (provided by Azure ML) where results CSV should be saved.')


    args = parser.parse_args()

    print("--- Script Execution Started ---")
    print(f"Azure ML Input - Energy Data Path: {args.energy_data}")
    print(f"Azure ML Input - Onshore Weather Path: {args.onshore_weather}")
    print(f"Azure ML Input - Offshore Weather Path: {args.offshore_weather}")
    print(f"Azure ML Output - Output Directory: {args.output_dir}")

    #  Configuration for Random Search 
    N_RANDOM_ITERATIONS = 50 
    # Define results filename - it will be saved INSIDE args.output_dir
    RESULTS_FILENAME = f'random_search_results_{N_RANDOM_ITERATIONS}_iters.csv'
    output_csv_file = Path(args.output_dir) / RESULTS_FILENAME # Full path for saving results
    print(f"Results will be saved to: {output_csv_file}")

    #  Define Hyperparameter Search Space 
    # These are the parameters varied in each call to run_experiment

    param_grid_for_random = {
    #  Parameters BEING TUNED (Focused Range)
    'sequence_length': [12, 18, 24],                  
    'lag_hours': [1],                                 
    'scaler_type': ['RobustScaler', 'StandardScaler'], 
    'tcn_num_filters': [12, 16, 24, 32],              
    'tcn_kernel_size': [6, 9, 12],                    
    'tcn_dropout_rate': [0.15, 0.2, 0.25, 0.3, 0.35],  
    'l2_factor': [0.0, 1e-4, 5e-4, 1e-3, 2e-3, 5e-3],  
    'learning_rate': [5e-4, 1e-3, 1.5e-3],             
    'batch_size': [64, 128],                          

    #  Parameters FIXED during this search                    
    'start_date': ['2020-01-01'],
    'validation_split_frac': [0.2],
    'tcn_dilations': [(1, 2, 4, 8)],                  
    'tcn_num_stacks': [1],                           
    'padding': ['causal'],
    'use_skip_connections': [True],
    'return_sequences': [False],                     # MUST be False
    'activation': ['relu'],
    'epochs': [75],                                  
    'use_lr_scheduler': [True]
    }

    #  Setup Results Tracking 
    results_list = []
    start_tuning_time = time.time()

    from sklearn.model_selection import ParameterGrid
    all_combinations_list = list(ParameterGrid(param_grid_for_random))
    total_possible = len(all_combinations_list)

    n_iterations = min(N_RANDOM_ITERATIONS, total_possible)
    if n_iterations < N_RANDOM_ITERATIONS:
         print(f"Warning: Requested {N_RANDOM_ITERATIONS} iterations, but only {total_possible} unique combinations exist. Running {n_iterations}.")

    print(f"Starting Random Search: Running {n_iterations} iterations from {total_possible} possible combinations...")

    sampled_indices = random.sample(range(total_possible), n_iterations)
    sampled_grid = [all_combinations_list[i] for i in sampled_indices]

    #  Iterate Through RANDOMLY SAMPLED Parameter Grid with TQDM 
    for i, params in enumerate(tqdm(sampled_grid, total=n_iterations, desc="Random Search Progress")):

        # Handle conditional parameters
        params['use_l2_reg'] = params.get('l2_factor', 0.0) > 0

        #  Call run_experiment, passing data paths and output dir from args 
        val_mae, val_rmse = run_experiment(
            energy_data_path=args.energy_data,          # Pass path from command line arg
            onshore_weather_path=args.onshore_weather,  # Pass path from command line arg
            offshore_weather_path=args.offshore_weather,# Pass path from command line arg
            output_dir=args.output_dir,                 # Pass path from command line arg
            **params,                                   # Pass hyperparams from the grid
            verbose=1                                   # Control verbosity of experiment run
        )

        # Store results
        result_record = params.copy()
        result_record['validation_mae_actual'] = val_mae
        result_record['validation_rmse_actual'] = val_rmse
        result_record['iteration'] = i + 1 # Track iteration number
        results_list.append(result_record)

        # Optional: Save intermediate results periodically
        current_iteration = len(results_list)
        if current_iteration % 10 == 0 or current_iteration == n_iterations:
            try:
                temp_df = pd.DataFrame(results_list)
                # Save to the full path defined using args.output_dir
                temp_df.to_csv(output_csv_file, index=False)
                print(f"\nIntermediate results saved to '{output_csv_file}' after iteration {current_iteration}.")
            except Exception as e:
                print(f"\nWarning: Failed to save intermediate results: {e}")


    #  Analyze Results 
    print("\n--- Tuning Complete ---")
    end_tuning_time = time.time()
    print(f"Total Tuning Time: {(end_tuning_time - start_tuning_time)/60:.2f} minutes for {n_iterations} iterations")

    if not results_list:
        print("No experiments completed successfully.")
    else:
        results_df = pd.DataFrame(results_list)
        # Filter out failed runs before sorting if any exist
        results_df = results_df[results_df['validation_mae_actual'] != np.inf]
        results_df = results_df.sort_values(by='validation_mae_actual', ascending=True) # Sort by actual MAE

        try:
            # Save final results to the full path defined using args.output_dir
            results_df.to_csv(output_csv_file, index=False)
            print(f"\nFinal results saved to '{output_csv_file}'")
        except Exception as e:
             print(f"Warning: Failed to save final results: {e}")


        print("\n--- Top 5 Best Parameter Combinations (by Actual Validation MAE) ---")
        # Select relevant columns for display
        display_cols = ['iteration', 'validation_mae_actual', 'validation_rmse_actual']
        # Add hyperparameter keys that were actually varied
        for key in param_grid_for_random:
             if isinstance(param_grid_for_random.get(key), list) and len(param_grid_for_random.get(key, [])) > 1:
                  display_cols.append(key)
        # Ensure columns exist in the dataframe before trying to display
        display_cols = [col for col in display_cols if col in results_df.columns]
        print(results_df.head(5)[display_cols].to_string())

        # Print best parameters clearly
        if not results_df.empty:
            best_run = results_df.iloc[0]
            if not best_run.isnull().all():
                best_mae = best_run['validation_mae_actual']
                best_rmse = best_run['validation_rmse_actual']
                print("\n--- Best Parameters Found ---")
                # Filter out performance metrics and iteration number for clarity
                best_params_dict = best_run.drop(['validation_mae_actual', 'validation_rmse_actual', 'iteration'], errors='ignore').to_dict()
                for key, value in best_params_dict.items():
                    # Only show parameters that were part of the search grid and varied
                     if key in param_grid_for_random and isinstance(param_grid_for_random.get(key), list) and len(param_grid_for_random.get(key, [])) > 1:
                        print(f"{key}: {value}")
                     elif key in ['use_l2_reg']: # Always show this conditional param
                          print(f"{key}: {value}")

                print(f"\nBest Actual Validation MAE: {best_mae:.4f}")
                print(f"Best Actual Validation RMSE: {best_rmse:.4f}")
            else:
                 print("Best run data seems invalid (contains NaNs).")
        else:
             print("No successful runs completed to determine best parameters.")

    print("--- Script Execution Finished ---")