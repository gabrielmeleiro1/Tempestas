import numpy as np
import pandas as pd
import joblib
import os
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler, RobustScaler

# Import custom modules
import config
from data_processing import prepare_data
from model_definition import create_tcn_regression_model
from training_utils import create_sequences
from evaluation import (
    evaluate_model_actual_scale,
    calculate_baselines,
    print_summary_comparison,
    plot_training_history,
    plot_predictions_vs_actual,
    print_config_summary
)

def main():
    # --- 1. Data Preparation ---
    print("--- 1. Preparing Data ---")
    X, y, final_df_processed = prepare_data()
    NUM_FEATURES = X.shape[1]
    NUM_OUTPUTS = y.shape[1]
    print(f"Data preparation complete. Features: {NUM_FEATURES}, Outputs: {NUM_OUTPUTS}")

    # Dynamically update filenames based on feature count
    scaler_x_path = f"scaler_x_{NUM_FEATURES}feat_{config.SCALER_TYPE}.joblib"
    scaler_y_path = f"scaler_y_{NUM_FEATURES}feat_{config.SCALER_TYPE}.joblib"
    model_filename = f'best_tcn_model_{NUM_FEATURES}feat_reg.keras'


    # --- 2. Train/Test Split ---
    print("\n--- 2. Splitting Data ---")
    num_samples = len(X)
    split_index = int(num_samples * (1 - config.VALIDATION_SPLIT_FRAC))
    X_train_df, y_train_df = X[:split_index], y[:split_index]
    X_test_df, y_test_df = X[split_index:], y[split_index:]
    print(f"Train shapes: X={X_train_df.shape}, y={y_train_df.shape}")
    print(f"Test shapes:  X={X_test_df.shape}, y={y_test_df.shape}")
    if not X_train_df.empty and not X_test_df.empty:
        assert X_train_df.index.max() < X_test_df.index.min(), "Train/Test split overlap detected!"

    # --- 3. Handle Lag NaNs in Training Data ---
    lag_cols_in_train = [col for col in X_train_df.columns if f'_Lag{config.LAG_HOURS}H' in col]
    if lag_cols_in_train and X_train_df[lag_cols_in_train].isnull().any().any():
        print(f"\n--- 3. Handling NaNs in Lagged Training Features ---")
        original_train_len = len(X_train_df)
        valid_index = X_train_df.dropna(subset=lag_cols_in_train).index
        X_train_df = X_train_df.loc[valid_index]
        y_train_df = y_train_df.loc[valid_index]
        print(f"Removed {original_train_len - len(X_train_df)} initial rows from training set.")
    else:
        print("\n--- 3. No Lag NaNs to handle in Training Data ---")


    # --- 4. Scaling Data ---
    print(f"\n--- 4. Scaling Data (using {config.SCALER_TYPE}) ---")
    if config.SCALER_TYPE == 'RobustScaler':
        scaler_x, scaler_y = RobustScaler(), RobustScaler()
    else:
        scaler_x, scaler_y = StandardScaler(), StandardScaler()

    X_train_scaled = scaler_x.fit_transform(X_train_df)
    y_train_scaled = scaler_y.fit_transform(y_train_df)
    X_test_scaled = scaler_x.transform(X_test_df)
    y_test_scaled = scaler_y.transform(y_test_df)
    print("Scaling complete.")
    print(f"Scaled Shapes: Train X={X_train_scaled.shape}, y={y_train_scaled.shape} | Test X={X_test_scaled.shape}, y={y_test_scaled.shape}")

    joblib.dump(scaler_x, scaler_x_path)
    joblib.dump(scaler_y, scaler_y_path)
    print(f"Fitted scalers saved to {scaler_x_path} and {scaler_y_path}")


    # --- 5. Creating Sequences ---
    print("\n--- 5. Creating Sequences ---")
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, config.SEQUENCE_LENGTH)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, config.SEQUENCE_LENGTH)

    if X_train_seq.size == 0 or X_test_seq.size == 0:
        print("Error: Sequence creation failed. Exiting.")
        return # Exit if no sequences could be made

    print(f"\nSequence shapes: "
          f"X_train={X_train_seq.shape}, y_train={y_train_seq.shape}, "
          f"X_test={X_test_seq.shape}, y_test={y_test_seq.shape}")


    # --- 6. Model Definition ---
    print("\n--- 6. Defining TCN Model ---")
    tcn_model = create_tcn_regression_model(
        input_shape=(config.SEQUENCE_LENGTH, NUM_FEATURES),
        num_outputs=NUM_OUTPUTS,
        nb_filters=config.TCN_NUM_FILTERS,
        kernel_size=config.TCN_KERNEL_SIZE,
        dilations=config.TCN_DILATIONS,
        nb_stacks=config.TCN_NUM_STACKS,
        padding=config.PADDING,
        use_skip_connections=config.USE_SKIP_CONNECTIONS,
        return_sequences=config.RETURN_SEQUENCES,
        dropout_rate=config.TCN_DROPOUT_RATE,
        activation=config.ACTIVATION,
        kernel_initializer=config.KERNEL_INITIALIZER,
        use_batch_norm=config.USE_BATCH_NORM,
        use_layer_norm=config.USE_LAYER_NORM,
        use_l2_reg=config.USE_L2_REG,
        l2_factor=config.L2_FACTOR,
        opt='adam',
        lr=config.LEARNING_RATE,
        model_name=f'TCN_{NUM_FEATURES}Feat_Reg'
    )
    tcn_model.summary()


    # --- 7. Model Training ---
    print("\n--- 7. Training Model ---")
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
    model_checkpoint = ModelCheckpoint(filepath=model_filename, monitor='val_loss', save_best_only=True, verbose=1)
    callbacks_list = [early_stopping, model_checkpoint]

    if config.USE_LR_SCHEDULER:
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=4, min_lr=0.00001, verbose=1)
        callbacks_list.append(reduce_lr)
        print("Using ReduceLROnPlateau scheduler.")

    fit_verbose = 1 # Default Keras verbosity
    try:
        from tqdm.keras import TqdmCallback
        tqdm_callback = TqdmCallback(verbose=1)
        fit_verbose = 0 # Use TQDM progress bar instead
        callbacks_list.append(tqdm_callback)
        print("Using TqdmCallback for progress.")
    except ImportError:
        print("TqdmCallback not found, using standard Keras progress bar.")

    history = tcn_model.fit(
        X_train_seq,
        y_train_seq,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        validation_data=(X_test_seq, y_test_seq),
        callbacks=callbacks_list,
        verbose=fit_verbose
    )
    print(f"Best model saved to {model_filename}")


    # --- 8. Evaluation and Prediction ---
    print("\n--- 8. Evaluating Model and Making Predictions ---")
    # Load the best model saved by ModelCheckpoint
    best_model = tf.keras.models.load_model(model_filename, custom_objects={'TCN': TCN}) # Need custom_objects for TCN layer
    print("Loaded best model for evaluation.")

    scaled_evaluation = best_model.evaluate(X_test_seq, y_test_seq, batch_size=config.BATCH_SIZE * 2, verbose=0)
    print(f"Scaled Normalized Test Loss (MSE): {scaled_evaluation[0]:.4f}")
    if len(scaled_evaluation) > 1: print(f"Scaled Normalized Test MAE: {scaled_evaluation[1]:.4f}")

    scaled_normalized_predictions = best_model.predict(X_test_seq, batch_size=config.BATCH_SIZE * 2)

    # --- Inverse Transform ---
    print("Inverse transforming predictions...")
    # 1. Inverse scale -> Normalized predictions
    normalized_predictions = scaler_y.inverse_transform(scaled_normalized_predictions)

    # 2. Get corresponding capacity proxy values for the test sequences
    capacity_proxy_test_period = final_df_processed.loc[X_test_df.index, 'capacity_proxy']
    capacity_proxy_test_seq = []
    # Ensure index alignment for proxy lookup
    num_test_sequences = len(X_test_seq)
    original_test_indices = X_test_df.index
    for i in range(num_test_sequences):
         # Get the index corresponding to the *end* of the i-th test sequence
         end_of_sequence_index_loc = i + config.SEQUENCE_LENGTH -1
         if end_of_sequence_index_loc < len(original_test_indices):
             target_timestamp = original_test_indices[end_of_sequence_index_loc]
             capacity_proxy_test_seq.append(final_df_processed.loc[target_timestamp, 'capacity_proxy'])
         else:
             print(f"Warning: Index out of bounds when looking up capacity proxy for sequence {i}. Appending NaN.")
             capacity_proxy_test_seq.append(np.nan) # Or handle differently

    capacity_proxy_test_seq = np.array(capacity_proxy_test_seq).reshape(-1, 1)

    # Handle potential length mismatch or NaNs from proxy lookup
    valid_proxy_indices = ~np.isnan(capacity_proxy_test_seq).flatten()
    if len(normalized_predictions) != len(capacity_proxy_test_seq):
         min_len = min(len(normalized_predictions), len(capacity_proxy_test_seq))
         print(f"Warning: Mismatch pred ({len(normalized_predictions)}) vs proxy ({len(capacity_proxy_test_seq)}). Truncating to {min_len}.")
         normalized_predictions = normalized_predictions[:min_len]
         capacity_proxy_test_seq = capacity_proxy_test_seq[:min_len]
         valid_proxy_indices = valid_proxy_indices[:min_len] # Adjust valid indices too

    normalized_predictions = normalized_predictions[valid_proxy_indices]
    capacity_proxy_test_seq = capacity_proxy_test_seq[valid_proxy_indices]
    y_test_seq_filtered = y_test_seq[valid_proxy_indices] # Filter corresponding true values

    if len(normalized_predictions) == 0:
        print("Error: No valid predictions remain after aligning with capacity proxy. Cannot evaluate.")
        return

    # 3. Convert NORMALIZED predictions -> ACTUAL SCALE (MW) predictions
    actual_scale_predictions = normalized_predictions * capacity_proxy_test_seq

    # 4. Get ORIGINAL actual scale targets for comparison
    normalized_y_test = scaler_y.inverse_transform(y_test_seq_filtered)
    actual_scale_y_test = normalized_y_test * capacity_proxy_test_seq

    # --- Actual Scale Evaluation ---
    tcn_results = evaluate_model_actual_scale(actual_scale_y_test, actual_scale_predictions)


    # --- 9. Plotting Results ---
    print("\n--- 9. Plotting Results ---")
    if config.PLOT_LEARNING_CURVES:
        plot_training_history(history)
    if config.PLOT_PREDICTIONS:
        plot_predictions_vs_actual(actual_scale_y_test, actual_scale_predictions, config.PLOT_PREDICTIONS_SLICE_LEN)


    # --- 10. Baseline Comparison & Summary ---
    print("\n--- 10. Baseline Comparison & Summary ---")
    baseline_results = calculate_baselines(
        final_df_processed, X_test_df, actual_scale_y_test, config.SEQUENCE_LENGTH, X_train_df
    )
    print_summary_comparison(tcn_results, baseline_results)


    # --- 11. Print Configuration ---
    print_config_summary() # Print the config used for this run

    print("\n--- Training and Evaluation Complete ---")


if __name__ == "__main__":
    # Ensure TensorFlow logging is set appropriately
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Suppress INFO messages
    # tf.get_logger().setLevel('WARNING')

    main()