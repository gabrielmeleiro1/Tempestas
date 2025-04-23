# XGBoost Price Prediction Script with Optuna Tuning 

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns 
from pathlib import Path
import optuna
import traceback 

current_dir = Path.cwd() 
base_dir = current_dir.parent 


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

#  Base Model Parameters
BASE_XGB_PARAMS = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'n_estimators': 2000,          # Use a high value, rely on early stopping
    'early_stopping_rounds': 50,
    'n_jobs': -1,
    'random_state': 99
    #we don't need to define anything else - everything else is handled by optuna
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

# Load Data 
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


# Merge Data 
merged_df = stage1_preds_df.merge(price_df, left_index=True, right_index=True, how='inner')
print(f"  Shape after merging: {merged_df.shape}")
merged_df.dropna(inplace=True) # Drop rows if any mismatch occurred (shouldn't with inner)
merged_df.sort_index(inplace=True)
print(f"  Shape after initial dropna: {merged_df.shape}")


# Feature Engineering 
features_df = merged_df.copy()
# Time Features
features_df['hour'] = features_df.index.hour
features_df['dayofweek'] = features_df.index.dayofweek
features_df['dayofyear'] = features_df.index.dayofyear
features_df['month'] = features_df.index.month
features_df['year'] = features_df.index.year
features_df['weekofyear'] = features_df.index.isocalendar().week.astype(int)
# Lagged Price Features
for lag in PRICE_LAGS:
    features_df[f'price_lag_{lag}h'] = features_df[TARGET_COL].shift(lag)
print(f"  Created {len(PRICE_LAGS)} lagged price features.")
# Handle nans created by lagging
features_df.dropna(inplace=True)
print(f"  Shape after feature engineering and NaN drop: {features_df.shape}")


# Data Splitting (Train/Validation/Test) 
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


# Define features and targets
print("\n[ Defining Features and Target]")
feature_cols = [col for col in features_df.columns if col != TARGET_COL]
X_train = train_df[feature_cols]
y_train = train_df[TARGET_COL]
X_val = val_df[feature_cols]
y_val = val_df[TARGET_COL]
X_test = test_df[feature_cols]
y_test = test_df[TARGET_COL]
print(f"  X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")


#  objective function definition

def objective(trial):
    """Optuna objective function to minimize validation RMSE (Refined Space)."""
    params = {
        #  Static Parameters already defined
        'objective': BASE_XGB_PARAMS['objective'],
        'eval_metric': BASE_XGB_PARAMS['eval_metric'],
        'n_jobs': BASE_XGB_PARAMS['n_jobs'],
        'random_state': BASE_XGB_PARAMS['random_state'],
        'early_stopping_rounds': BASE_XGB_PARAMS['early_stopping_rounds'],
        'n_estimators': BASE_XGB_PARAMS['n_estimators'], # Use fixed high n_estimators

        #  Tunable Parameters - based on a previous test, we try to further optimize it 
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.08, log=True),

        'max_depth': trial.suggest_int('max_depth', 5, 10),

        'subsample': trial.suggest_float('subsample', 0.6, 1.1, step=0.1),

        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.1, step=0.1),

        'gamma': trial.suggest_float('gamma', 2.0, 5.5),
        
        'reg_alpha': trial.suggest_float('reg_alpha', 0.05, 1.0, log=True),

        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 0.1, log=True),

        'min_child_weight': trial.suggest_int('min_child_weight', 2, 8),
    }

    model = xgb.XGBRegressor(**params)
    eval_set_optuna = [(X_train, y_train), (X_val, y_val)] # Evaluate against validation set

    try:
        model.fit(X_train, y_train, eval_set=eval_set_optuna, verbose=False) 
        results = model.evals_result()
        # Get RMSE from the validation set ('validation_1') at the best iteration found by early stopping
        validation_rmse = results['validation_1']['rmse'][model.best_iteration]
    except Exception as e:
        print(f" Trial failed with error: {e}. Returning high RMSE.")
        # Return a large value if the trial fails (e.g., due to invalid parameters)
        return float('inf')

    return validation_rmse

print("  Objective function defined.")


# optuna run
print("\n[Running Optuna Study]")
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=N_OPTUNA_TRIALS)

best_params = study.best_params
best_value = study.best_value
print(f"  Optuna study finished after {len(study.trials)} trials.")
print(f"  Best validation RMSE found: {best_value:.4f}")
print(f"  Best parameters: {best_params}")

# Create final parameter set
final_xgb_params = BASE_XGB_PARAMS.copy()
final_xgb_params.update(best_params) 


# train the final model
print("\n[Training Final Model with Best Parameters]")
# Use the full training set (X_train, y_train)
# Use early stopping against the validation set (X_val, y_val) to determine optimal n_estimators
final_model = xgb.XGBRegressor(**final_xgb_params)
final_eval_set = [(X_train, y_train), (X_val, y_val)] # Stop based on validation performance

final_model.fit(
    X_train, y_train,
    eval_set=final_eval_set,
    verbose=100 # to show progress during final training
)
print(f"  Final model training completed. Best iteration: {final_model.best_iteration}")

# Save the model
print(f"  Saving final model to {FINAL_MODEL_SAVE_PATH}...")
final_model.save_model(FINAL_MODEL_SAVE_PATH)
print("  Model saved.")


#  Evaluate Final Model on Test Set 
print("\n[Evaluating Final Model on Test Set]")
y_pred_final = final_model.predict(X_test)
mae_final = mean_absolute_error(y_test, y_pred_final)
rmse_final = np.sqrt(mean_squared_error(y_test, y_pred_final))

print("  --- Final Model Performance (Test Set) ---")
print(f"    MAE:  {mae_final:.3f} EUR/MWh")
print(f"    RMSE: {rmse_final:.3f} EUR/MWh")


# Baseline Calculation (on Test Set) 
print("\n[Calculating Baselines (Test Set)]")
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


# Feature importance
# print("\n[Feature Importance (Final Model)]")
# importance_scores = final_model.get_booster().get_score(importance_type='weight')
# importance_df = pd.DataFrame({
#     'Feature': importance_scores.keys(),
#     'Importance': importance_scores.values()
# }).sort_values('Importance', ascending=False).reset_index(drop=True)
# print("  Top 15 Features by Importance (Weight):")
# print(importance_df.head(15).to_string()) # Print df as string for better script output

print("\n--- Script Completed Successfully ---")