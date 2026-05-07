# XGBoost Price Prediction Script with Optuna Tuning 

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns 
from pathlib import Path
import sys
import optuna
import traceback 

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from wind_power_prediction.feature_schema import price_feature_columns

# 1. Configuration


current_dir = Path(__file__).resolve().parent
base_dir = PROJECT_ROOT


stage1_start_date_config = '2020-01-01' # The start_date used for Stage 1 training/prediction
STAGE1_PREDICTIONS_FILENAME = f'stage1_predictions_mw_model_trained_from_{stage1_start_date_config}.csv'
STAGE1_PREDICTIONS_PATH = current_dir / STAGE1_PREDICTIONS_FILENAME



PRICE_DATA_PATH = base_dir / 'datasets' / 'energy_price' / 'nl_wholesale_electricity_price_data_hourly.csv'


FINAL_MODEL_SAVE_PATH = current_dir / 'final_xgboost_price_model_tuned.json' 

# Feature Engineering 
TARGET_COL = 'Price (EUR/MWhe)' # Ensure this matches the price CSV column name EXACTLY
STAGE1_FEATURES_TO_USE = ['Predicted_Offshore_MW', 'Predicted_Onshore_MW']
PRICE_LAGS = [1, 2, 3, 6, 12, 24] # Lags in hours

# Data Splitting 
TEST_SET_DAYS = 30
VAL_SET_DAYS = 30 # Use another 30 days for validation during tuning
TEST_SET_PERIODS = TEST_SET_DAYS * 24
VAL_SET_PERIODS = VAL_SET_DAYS * 24

# Optuna Tuning
N_OPTUNA_TRIALS = 100 # Number of tuning trials 

#  Base Model Parameters (defaults, some will be tuned)
BASE_XGB_PARAMS = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'n_estimators': 2000,          # Use a high value, rely on early stopping
    'early_stopping_rounds': 50,
    'n_jobs': -1,
    'random_state': 99
    # Other parameters like learning_rate, max_depth, etc., will be suggested by Optuna
}

#  Print configuration 
print(f"  Current Directory:  {current_dir.resolve()}")
print(f"  Base Directory:     {base_dir.resolve()}")
print(f"  Stage 1 Preds Path: {STAGE1_PREDICTIONS_PATH.resolve()}")
print(f"  Price Data Path:    {PRICE_DATA_PATH.resolve()}")
print(f"  Final Model Path:   {FINAL_MODEL_SAVE_PATH.resolve()}") # Added print for model save path
print(f"  Target Column:      {TARGET_COL}")
print(f"  Val/Test Periods:   {VAL_SET_PERIODS} / {TEST_SET_PERIODS}")
print(f"  Optuna Trials:      {N_OPTUNA_TRIALS}")

#  2. Load Data
print("\n[2. Loading Data]")
# Load Stage 1 Predictions
stage1_preds_df = pd.read_csv(STAGE1_PREDICTIONS_PATH, index_col='Timestamp (UTC)', parse_dates=True)
if stage1_preds_df.index.tz is None: stage1_preds_df.index = stage1_preds_df.index.tz_localize('UTC')
else: stage1_preds_df.index = stage1_preds_df.index.tz_convert('UTC')
print(f"  Loaded Stage 1 predictions: {stage1_preds_df.shape}")

# Load Price Data
price_df = pd.read_csv(PRICE_DATA_PATH, index_col='Datetime (UTC)', parse_dates=True)
if price_df.index.tz is None: price_df.index = price_df.index.tz_localize('UTC')
else: price_df.index = price_df.index.tz_convert('UTC')
price_df = price_df[[TARGET_COL]] # Select only the target price column
print(f"  Loaded Price data: {price_df.shape}")


# --- 3. Merge Data ---
print("\n[3. Merging Data]")
merged_df = stage1_preds_df.merge(price_df, left_index=True, right_index=True, how='inner')
print(f"  Shape after merging: {merged_df.shape}")
merged_df.dropna(inplace=True) # Drop rows if any mismatch occurred (shouldn't with inner)
merged_df.sort_index(inplace=True)
print(f"  Shape after initial dropna: {merged_df.shape}")


# --- 4. Feature Engineering ---
print("\n[4. Feature Engineering]")
features_df = merged_df.copy()
# Time Features
features_df['hour'] = features_df.index.hour
features_df['dayofweek'] = features_df.index.dayofweek
features_df['dayofyear'] = features_df.index.dayofyear
features_df['month'] = features_df.index.month
features_df['year'] = features_df.index.year
features_df['weekofyear'] = features_df.index.isocalendar().week.astype(int)
print("  Created time features.")
# Lagged Price Features
for lag in PRICE_LAGS:
    features_df[f'price_lag_{lag}h'] = features_df[TARGET_COL].shift(lag)
print(f"  Created {len(PRICE_LAGS)} lagged price features.")
# Handle NaNs created by lagging
features_df.dropna(inplace=True)
print(f"  Shape after feature engineering and NaN drop: {features_df.shape}")


# --- 5. Data Splitting (Train/Validation/Test) ---
print("\n[5. Splitting Data]")
n_total_samples = len(features_df)
n_test_samples = TEST_SET_PERIODS
n_val_samples = VAL_SET_PERIODS
n_train_samples = n_total_samples - n_val_samples - n_test_samples

if n_train_samples <= 0 or n_val_samples <= 0 or n_test_samples <= 0:
    raise ValueError(f"Data sizes are not positive after splitting. Total: {n_total_samples}, Train: {n_train_samples}, Val: {n_val_samples}, Test: {n_test_samples}.")

val_start_index = n_train_samples
test_start_index = n_train_samples + n_val_samples

train_df = features_df.iloc[:val_start_index]
val_df = features_df.iloc[val_start_index:test_start_index]
test_df = features_df.iloc[test_start_index:]

print(f"  Training set shape:   {train_df.shape} ({train_df.index.min()} to {train_df.index.max()})")
print(f"  Validation set shape: {val_df.shape} ({val_df.index.min()} to {val_df.index.max()})")
print(f"  Testing set shape:    {test_df.shape} ({test_df.index.min()} to {test_df.index.max()})")


# --- 6. Define Features (X) and Target (y) ---
print("\n[6. Defining Features and Target]")
feature_cols = price_feature_columns(PRICE_LAGS)
X_train = train_df[feature_cols]
y_train = train_df[TARGET_COL]
X_val = val_df[feature_cols]
y_val = val_df[TARGET_COL]
X_test = test_df[feature_cols]
y_test = test_df[TARGET_COL]
print(f"  X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")


#  7. Optuna Objective Function Definition (REFINED SEARCH SPACE)
print("\n[7. Defining Optuna Objective with Refined Search Space]")

def objective(trial):
    """Optuna objective function to minimize validation RMSE (Refined Space)."""
    params = {
        #  Static Parameters (from BASE_XGB_PARAMS)
        'objective': BASE_XGB_PARAMS['objective'],
        'eval_metric': BASE_XGB_PARAMS['eval_metric'],
        'n_jobs': BASE_XGB_PARAMS['n_jobs'],
        'random_state': BASE_XGB_PARAMS['random_state'],
        'early_stopping_rounds': BASE_XGB_PARAMS['early_stopping_rounds'],
        'n_estimators': BASE_XGB_PARAMS['n_estimators'], # Use fixed high n_estimators

        #  Tunable Parameters (Ranges adjusted based on previous best results)

        # Previous best: 0.0307. Narrowing the range, focusing below 0.1.
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.08, log=True),

        # Previous best: 7. Range (3, 10) was okay. Slightly narrowing to (5, 10).
        'max_depth': trial.suggest_int('max_depth', 5, 10),

        # Previous best: 0.8. Range (0.6, 1.0) seems good. Keeping it or slightly narrowing.
        'subsample': trial.suggest_float('subsample', 0.6, 1.0, step=0.1),

        # Previous best: 1.0 (edge). Range (0.6, 1.0). Focusing on higher values.
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0, step=0.1),

        # Previous best: 3.73. Range (0, 5). Focusing on the upper part of the previous range.
        'gamma': trial.suggest_float('gamma', 2.0, 5.5),

        # Previous best: 0.2655. Range (1e-8, 1.0). Narrowing slightly from below.
        'reg_alpha': trial.suggest_float('reg_alpha', 0.05, 1.0, log=True),

        # Previous best: ~6e-6 (very small). Range (1e-8, 5.0). Drastically reducing upper bound.
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 0.1, log=True),

        # Previous best: 4. Range (1, 10). Narrowing the range around the best value.
        'min_child_weight': trial.suggest_int('min_child_weight', 2, 8),
    }

    model = xgb.XGBRegressor(**params)
    eval_set_optuna = [(X_train, y_train), (X_val, y_val)] # Evaluate against validation set

    try:
        model.fit(X_train, y_train, eval_set=eval_set_optuna, verbose=False) # verbose=False for cleaner Optuna logs
        results = model.evals_result()
        # Get RMSE from the validation set ('validation_1') at the best iteration found by early stopping
        validation_rmse = results['validation_1']['rmse'][model.best_iteration]
    except Exception as e:
        print(f"!!! Trial failed with error: {e}. Returning high RMSE.")
        # Return a large value if the trial fails (e.g., due to invalid parameters)
        # You might want to prune the trial instead using `raise optuna.exceptions.TrialPruned()`
        # depending on how you want Optuna to handle failures.
        return float('inf') # Or some other large number


    return validation_rmse

print("  Objective function defined.")


# --- 8. Run Optuna Study ---
print("\n[8. Running Optuna Study]")
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=N_OPTUNA_TRIALS)

best_params = study.best_params
best_value = study.best_value
print(f"  Optuna study finished after {len(study.trials)} trials.")
print(f"  Best validation RMSE found: {best_value:.4f}")
print(f"  Best parameters: {best_params}")

# Create final parameter set
final_xgb_params = BASE_XGB_PARAMS.copy()
final_xgb_params.update(best_params) # Update with tuned values


# --- 9. Train Final Model ---
print("\n[9. Training Final Model with Best Parameters]")
# Use the full training set (X_train, y_train)
# Use early stopping against the validation set (X_val, y_val) to determine optimal n_estimators
final_model = xgb.XGBRegressor(**final_xgb_params)
final_eval_set = [(X_train, y_train), (X_val, y_val)] # Stop based on validation performance

final_model.fit(
    X_train, y_train,
    eval_set=final_eval_set,
    verbose=100 # Show progress during final training
)
print(f"  Final model training completed. Best iteration: {final_model.best_iteration}")

# Save the trained model
print(f"  Saving final model to {FINAL_MODEL_SAVE_PATH}...")
final_model.save_model(FINAL_MODEL_SAVE_PATH)
print("  Model saved.")


# --- 10. Evaluate Final Model on Test Set ---
print("\n[10. Evaluating Final Model on Test Set]")
y_pred_final = final_model.predict(X_test)
mae_final = mean_absolute_error(y_test, y_pred_final)
rmse_final = np.sqrt(mean_squared_error(y_test, y_pred_final))

print("  --- Final Model Performance (Test Set) ---")
print(f"    MAE:  {mae_final:.3f} EUR/MWh")
print(f"    RMSE: {rmse_final:.3f} EUR/MWh")


# --- 11. Baseline Calculation (on Test Set) ---
print("\n[11. Calculating Baselines (Test Set)]")
# Persistence
persistence_lag_col = 'price_lag_1h'
y_pred_persistence = X_test[persistence_lag_col]
mae_persistence = mean_absolute_error(y_test, y_pred_persistence)
rmse_persistence = np.sqrt(mean_squared_error(y_test, y_pred_persistence))
print(f"  Persistence Baseline:")
print(f"    MAE:  {mae_persistence:.3f} EUR/MWh")
print(f"    RMSE: {rmse_persistence:.3f} EUR/MWh")
# Average
y_pred_average = np.full_like(y_test, fill_value=y_train.mean()) # Use train mean
mae_average = mean_absolute_error(y_test, y_pred_average)
rmse_average = np.sqrt(mean_squared_error(y_test, y_pred_average))
print(f"  Average Baseline:")
print(f"    MAE:  {mae_average:.3f} EUR/MWh")
print(f"    RMSE: {rmse_average:.3f} EUR/MWh")


# --- 12. Feature Importance (Optional Output) ---
print("\n[12. Feature Importance (Final Model)]")
importance_scores = final_model.get_booster().get_score(importance_type='weight')
importance_df = pd.DataFrame({
    'Feature': importance_scores.keys(),
    'Importance': importance_scores.values()
}).sort_values('Importance', ascending=False).reset_index(drop=True)
print("  Top 15 Features by Importance (Weight):")
print(importance_df.head(15).to_string()) # Print df as string for better script output


# --- 13. Completion ---
print("\n--- Script Completed Successfully ---")
