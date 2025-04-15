from pathlib import Path
import numpy as np

# Data Paths
BASE_DIR = Path(".")
ENERGY_CSV = BASE_DIR / 'combined_total_energy_data_2017_2025.csv'
ONSHORE_CSV = BASE_DIR / 'final_averaged_onshore_weather.csv'
OFFSHORE_CSV = BASE_DIR / 'final_averaged_offshore_weather.csv'

# Model/Sequence Parameters
SEQUENCE_LENGTH = 12
EPOCHS = 50
BATCH_SIZE = 128
VALIDATION_SPLIT_FRAC = 0.2
START_DATE = '2021-01-01' # Data filtering start date
CAPACITY_PROXY_WINDOW = '365D' # Rolling window for capacity proxy
LAG_HOURS = 1 # Hours for lagged target feature

# Scaling and Regularization
SCALER_TYPE = 'RobustScaler' # 'RobustScaler' or 'StandardScaler'
TCN_DROPOUT_RATE = 0.3
USE_L2_REG = True           # Flag to add L2 regularization to output layer
L2_FACTOR = 0.001           # L2 regularization factor

# TCN Architecture
TCN_NUM_FILTERS = 16
TCN_NUM_STACKS = 4
TCN_KERNEL_SIZE = 9
TCN_DILATIONS = [1, 2, 4, 8, 16]
PADDING = 'causal'
USE_SKIP_CONNECTIONS = True
RETURN_SEQUENCES = False
ACTIVATION = 'relu'
KERNEL_INITIALIZER = 'he_normal'
USE_BATCH_NORM = True
USE_LAYER_NORM = False

# Learning Rate Parameters
LEARNING_RATE = 0.001
USE_LR_SCHEDULER = True     # Flag to use ReduceLROnPlateau

# Feature Selection
COLS_TO_REMOVE = sorted(list(set([
    'hour_sin_onshore', 'hour_cos_onshore', 'day_of_week_sin_onshore', 'day_of_week_cos_onshore',
    'day_of_year_sin_onshore', 'day_of_year_cos_onshore', 'month_sin_offshore', 'month_cos_offshore',
    'month_sin_onshore', 'month_cos_onshore', 'wind_direction_10m_sin_offshore', 'wind_direction_10m_cos_offshore',
    'wind_direction_10m_sin_onshore', 'wind_direction_10m_cos_onshore', 'wind_speed_10m_offshore',
    'wind_gusts_10m_offshore', 'wind_speed_10m_onshore', 'wind_gusts_10m_onshore', 'surface_pressure_onshore',
    'cloud_cover_low_offshore', 'cloud_cover_mid_offshore', 'cloud_cover_high_offshore', 'cloud_cover_low_onshore',
    'cloud_cover_mid_onshore', 'cloud_cover_high_onshore'
])))

# Plotting
PLOT_CAPACITY_PROXY = True
PLOT_LEARNING_CURVES = True
PLOT_PREDICTIONS = True
PLOT_PREDICTIONS_SLICE_LEN = 500

# Baseline Evaluation
PERSISTENCE_LAGS = ['1H', '24H']
TARGET_COLS_ORIGINAL = ['Wind_Offshore_MW', 'Wind_Onshore_MW']

# Model/Scaler Saving
SCALER_X_FILENAME = f"scaler_x_{SCALER_TYPE}.joblib"
SCALER_Y_FILENAME = f"scaler_y_{SCALER_TYPE}.joblib"
MODEL_FILENAME = f'best_tcn_model_reg.keras' # Feature count will be added dynamically