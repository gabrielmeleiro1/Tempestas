import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import config # Import configuration

def evaluate_model_actual_scale(y_true_actual, y_pred_actual):
    mae_actual = mean_absolute_error(y_true_actual, y_pred_actual)
    mse_actual = mean_squared_error(y_true_actual, y_pred_actual)
    rmse_actual = np.sqrt(mse_actual)

    print(f"\n--- Performance on Actual Scale (Test Set) ---")
    print(f"Mean Absolute Error (MAE): {mae_actual:.2f} MW")
    print(f"Mean Squared Error (MSE):  {mse_actual:.2f} MW^2")
    print(f"Root Mean Squared Error (RMSE): {rmse_actual:.2f} MW")

    print("\nFirst 5 Actual Scale Predictions vs Targets:")
    original_target_names = ['Offshore', 'Onshore'] # Assumes this order
    for i in range(min(5, len(y_pred_actual))):
        pred_str = f"Pred {original_target_names[0]}={y_pred_actual[i, 0]:.1f}"
        actual_str = f"Actual {original_target_names[0]}={y_true_actual[i, 0]:.1f}"
        if y_pred_actual.shape[1] > 1:
            pred_str += f", Pred {original_target_names[1]}={y_pred_actual[i, 1]:.1f}"
            actual_str += f", Actual {original_target_names[1]}={y_true_actual[i, 1]:.1f}"
        print(f"Sample {i+1}: {pred_str} || {actual_str}")

    results = {'mae': mae_actual, 'rmse': rmse_actual}
    if y_true_actual.shape[1] > 1:
        results['mae_offshore'] = mean_absolute_error(y_true_actual[:, 0], y_pred_actual[:, 0])
        results['mae_onshore'] = mean_absolute_error(y_true_actual[:, 1], y_pred_actual[:, 1])
        results['rmse_offshore'] = np.sqrt(mean_squared_error(y_true_actual[:, 0], y_pred_actual[:, 0]))
        results['rmse_onshore'] = np.sqrt(mean_squared_error(y_true_actual[:, 1], y_pred_actual[:, 1]))
    return results


def calculate_baselines(final_df, X_test_df, y_test_actual_mw, sequence_length, X_train_df):
    print("\n--- Baseline Comparison ---")
    baseline_results = {}

    # --- Persistence Baseline ---
    try:
        test_set_indices = X_test_df.index
        target_indices = test_set_indices[sequence_length - 1 : len(test_set_indices)]
        if len(target_indices) != len(y_test_actual_mw):
             print(f"Warning: Baseline index length mismatch. Adjusting.")
             target_indices = target_indices[:len(y_test_actual_mw)] # Adjust index length

        y_test_actual_mw_for_baseline = final_df.loc[target_indices, config.TARGET_COLS_ORIGINAL]

        for lag in config.PERSISTENCE_LAGS:
            print(f"\nCalculating Persistence Baseline (Lag: {lag})...")
            try:
                start_time_needed = target_indices.min() - pd.Timedelta(lag)
                persistence_data = final_df.loc[start_time_needed:, config.TARGET_COLS_ORIGINAL]
                persistence_preds_shifted = persistence_data.shift(freq=lag)
                persistence_preds_aligned = persistence_preds_shifted.loc[target_indices]

                valid_indices = persistence_preds_aligned.dropna().index
                persistence_preds_final = persistence_preds_aligned.loc[valid_indices]
                y_test_actual_aligned = y_test_actual_mw_for_baseline.loc[valid_indices]

                if len(persistence_preds_final) == 0:
                    print(f"Error: No valid persistence predictions for lag {lag}.")
                    baseline_results[f'Persistence_{lag}'] = {'mae': np.nan, 'rmse': np.nan}
                    continue

                mae_persistence = mean_absolute_error(y_test_actual_aligned, persistence_preds_final)
                rmse_persistence = np.sqrt(mean_squared_error(y_test_actual_aligned, persistence_preds_final))
                print(f"Persistence ({lag}) MAE (MW): {mae_persistence:.2f}")
                print(f"Persistence ({lag}) RMSE (MW): {rmse_persistence:.2f}")
                baseline_results[f'Persistence_{lag}'] = {'mae': mae_persistence, 'rmse': rmse_persistence}

            except Exception as e:
                print(f"Error calculating persistence baseline for lag {lag}: {e}")
                baseline_results[f'Persistence_{lag}'] = {'mae': np.nan, 'rmse': np.nan}
    except Exception as e:
        print(f"An unexpected error occurred during persistence baseline setup: {e}")
        # Initialize keys to prevent errors later
        for lag in config.PERSISTENCE_LAGS:
             baseline_results[f'Persistence_{lag}'] = {'mae': np.nan, 'rmse': np.nan}


    # --- Simple Average Baseline ---
    print("\n--- Simple Average Model ---")
    if 'X_train_df' in locals() and not X_train_df.empty:
        train_set_indices = X_train_df.index
        try:
            train_targets_actual_mw = final_df.loc[train_set_indices, config.TARGET_COLS_ORIGINAL]
            train_mean_offshore = train_targets_actual_mw['Wind_Offshore_MW'].mean()
            train_mean_onshore = train_targets_actual_mw['Wind_Onshore_MW'].mean()

            if 'y_test_actual_mw_for_baseline' in locals() and not y_test_actual_mw_for_baseline.empty:
                avg_preds_offshore = np.full(len(y_test_actual_mw_for_baseline), train_mean_offshore)
                avg_preds_onshore = np.full(len(y_test_actual_mw_for_baseline), train_mean_onshore)
                avg_preds = np.stack([avg_preds_offshore, avg_preds_onshore], axis=1)

                mae_average = mean_absolute_error(y_test_actual_mw_for_baseline, avg_preds)
                rmse_average = np.sqrt(mean_squared_error(y_test_actual_mw_for_baseline, avg_preds))
                print(f"Simple Avg MAE (MW): {mae_average:.2f}")
                print(f"Simple Avg RMSE (MW): {rmse_average:.2f}")
                baseline_results['Simple_Average'] = {'mae': mae_average, 'rmse': rmse_average}
            else:
                 print("Error: y_test_actual_mw_for_baseline not available for Simple Average calculation.")
                 baseline_results['Simple_Average'] = {'mae': np.nan, 'rmse': np.nan}

        except Exception as e:
            print(f"Error calculating simple average baseline: {e}")
            baseline_results['Simple_Average'] = {'mae': np.nan, 'rmse': np.nan}
    else:
         print("X_train_df not found or empty. Skipping Simple Average baseline.")
         baseline_results['Simple_Average'] = {'mae': np.nan, 'rmse': np.nan}

    return baseline_results


def print_summary_comparison(tcn_results, baseline_results):
    print("\n--- D. Performance Summary ---")
    print(f"{'Metric':<15} | {'TCN Model':<15} | {'Persistence (1H)':<17} | {'Persistence (24H)':<18} | {'Simple Average':<15}")
    print(f"----------------|-----------------|-------------------|--------------------|-----------------")

    pers_1h_mae = baseline_results.get('Persistence_1H', {}).get('mae', np.nan)
    pers_1h_rmse = baseline_results.get('Persistence_1H', {}).get('rmse', np.nan)
    pers_24h_mae = baseline_results.get('Persistence_24H', {}).get('mae', np.nan)
    pers_24h_rmse = baseline_results.get('Persistence_24H', {}).get('rmse', np.nan)
    avg_mae = baseline_results.get('Simple_Average', {}).get('mae', np.nan)
    avg_rmse = baseline_results.get('Simple_Average', {}).get('rmse', np.nan)

    tcn_mae_overall = tcn_results.get('mae', np.nan)
    tcn_rmse_overall = tcn_results.get('rmse', np.nan)

    print(f"{'MAE (MW)':<15} | {tcn_mae_overall:<15.2f} | {pers_1h_mae:<17.2f} | {pers_24h_mae:<18.2f} | {avg_mae:<15.2f}")
    print(f"{'RMSE (MW)':<15} | {tcn_rmse_overall:<15.2f} | {pers_1h_rmse:<17.2f} | {pers_24h_rmse:<18.2f} | {avg_rmse:<15.2f}")

    if 'mae_offshore' in tcn_results:
         print(f"\n{'MAE Offshore':<15} | {tcn_results['mae_offshore']:.2f}")
         print(f"{'MAE Onshore':<15} | {tcn_results['mae_onshore']:.2f}")
         print(f"{'RMSE Offshore':<15} | {tcn_results['rmse_offshore']:.2f}")
         print(f"{'RMSE Onshore':<15} | {tcn_results['rmse_onshore']:.2f}")

def plot_training_history(history):
    if history and hasattr(history, 'history'):
        plt.figure(figsize=(12, 5))
        if 'loss' in history.history and 'val_loss' in history.history:
            plt.subplot(1, 2, 1)
            plt.plot(history.history['loss'], label='Training Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title('Model Loss (Scaled Normalized MSE)')
            plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)
            loss_max = max(history.history.get('val_loss', [0]))
            plt.ylim(bottom=0, top=min(loss_max * 1.2, 5) if loss_max > 0 else 1)

        if 'mae' in history.history and 'val_mae' in history.history:
            plt.subplot(1, 2, 2)
            plt.plot(history.history['mae'], label='Training MAE')
            plt.plot(history.history['val_mae'], label='Validation MAE')
            plt.title('Model MAE (Scaled Normalized)')
            plt.xlabel('Epoch'); plt.ylabel('MAE'); plt.legend(); plt.grid(True)
            mae_max = max(history.history.get('val_mae', [0]))
            plt.ylim(bottom=0, top=min(mae_max * 1.2, 2) if mae_max > 0 else 1)

        plt.tight_layout()
        plt.show()
    else:
        print("Training history not available for plotting.")

def plot_predictions_vs_actual(y_true_actual, y_pred_actual, slice_len):
    print("\nPlotting Actual vs Predicted (Actual MW Scale)...")
    plt.figure(figsize=(15, 6))
    plot_slice = slice(0, min(slice_len, len(y_true_actual)))

    if len(y_true_actual) > 0 and len(y_pred_actual) > 0 :
        if y_true_actual.shape[1] > 0:
            plt.plot(y_true_actual[plot_slice, 0], label='Actual Offshore MW', color='blue', alpha=0.7)
            plt.plot(y_pred_actual[plot_slice, 0], label='Predicted Offshore MW', color='red', linestyle='--')
        if y_true_actual.shape[1] > 1:
             plt.plot(y_true_actual[plot_slice, 1], label='Actual Onshore MW', color='cyan', alpha=0.7)
             plt.plot(y_pred_actual[plot_slice, 1], label='Predicted Onshore MW', color='magenta', linestyle=':')

        plt.title(f'Actual vs Predicted MW (Test Set Slice - First {plot_slice.stop} Points)')
        plt.xlabel('Time Steps (within slice)'); plt.ylabel('Power (MW)')
        plt.legend(); plt.grid(True); plt.show()
    else:
        print("No actual or predicted data available to plot.")

def print_config_summary():
    print("\n" + "="*30 + "\n=== Run Configuration Summary ===" + "\n" + "="*30)
    print(f"Start Date Filter:          {config.START_DATE}")
    print(f"Lagged Feature Hours (H):   {config.LAG_HOURS}")
    print(f"Validation Split Fraction:  {config.VALIDATION_SPLIT_FRAC}")
    print(f"Scaler Type:                {config.SCALER_TYPE}")
    # NUM_FEATURES/OUTPUTS printed during training setup
    print(f"Sequence Length:            {config.SEQUENCE_LENGTH}")
    print(f"Epochs:                     {config.EPOCHS}")
    print(f"Batch Size:                 {config.BATCH_SIZE}")
    print(f"Learning Rate:              {config.LEARNING_RATE}")
    print(f"Use LR Scheduler:           {config.USE_LR_SCHEDULER}")
    print(f"TCN Filters:                {config.TCN_NUM_FILTERS}")
    print(f"TCN Kernel Size:            {config.TCN_KERNEL_SIZE}")
    print(f"TCN Stacks:                 {config.TCN_NUM_STACKS}")
    print(f"TCN Dilations:              {config.TCN_DILATIONS}")
    print(f"TCN Padding:                {config.PADDING}")
    print(f"TCN Activation:             {config.ACTIVATION}")
    print(f"TCN Dropout Rate:           {config.TCN_DROPOUT_RATE}")
    print(f"TCN Use Skip Connections:   {config.USE_SKIP_CONNECTIONS}")
    print(f"TCN Return Sequences:       {config.RETURN_SEQUENCES}")
    print(f"Use L2 Regularization:      {config.USE_L2_REG}")
    print(f"L2 Factor:                  {config.L2_FACTOR if config.USE_L2_REG else 'N/A'}")
    print("="*30 + "\n")