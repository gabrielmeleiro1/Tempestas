import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import config # Import the configuration

def load_data(energy_path, onshore_path, offshore_path):
    energy_df = pd.read_csv(energy_path)
    onshore_weather_df = pd.read_csv(onshore_path)
    offshore_weather_df = pd.read_csv(offshore_path)
    return energy_df, onshore_weather_df, offshore_weather_df

def preprocess_and_merge(energy_df, onshore_weather_df, offshore_weather_df):
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
    print(f"Shape after merging: {final_df.shape}")

    final_df = final_df.sort_index()
    print(f"NaNs before fill: {final_df.isnull().sum().sum()}")
    final_df = final_df.ffill().bfill()
    print(f"NaNs after fill: {final_df.isnull().sum().sum()}")
    return final_df

def filter_data(df, start_date):
    print(f"\nOriginal shape (before date filter): {df.shape}")
    try:
        df_filtered = df[df.index >= start_date].copy() # Use .copy()
        print(f"Filtered shape (start date {start_date}): {df_filtered.shape}")
        if df_filtered.empty:
            raise ValueError(f"No data remaining after filtering for start date {start_date}.")
        return df_filtered
    except Exception as e:
        print(f"Error during date filtering: {e}")
        raise

def calculate_capacity_proxy_and_normalize(df, window, plot=True):
    df_processed = df.copy() # Work on a copy
    if 'Wind_Offshore_MW' in df_processed.columns and 'Wind_Onshore_MW' in df_processed.columns:
        df_processed['Total_Wind_MW'] = df_processed['Wind_Offshore_MW'] + df_processed['Wind_Onshore_MW']
        proxy_base_col = 'Total_Wind_MW'
    else:
        raise ValueError("Missing 'Wind_Offshore_MW' or 'Wind_Onshore_MW' for proxy calculation.")

    df_processed['capacity_proxy'] = df_processed[proxy_base_col].rolling(window=window, min_periods=1).max()
    df_processed['capacity_proxy'] = df_processed['capacity_proxy'].ffill().bfill()
    epsilon = 1e-6
    df_processed['capacity_proxy'] = df_processed['capacity_proxy'] + epsilon
    print(f"Capacity proxy calculated using {window} window.")

    target_cols_normalized = []
    if 'Wind_Offshore_MW' in df_processed.columns:
        df_processed['Offshore_Norm'] = (df_processed['Wind_Offshore_MW'] / df_processed['capacity_proxy']).clip(0, 1.1)
        target_cols_normalized.append('Offshore_Norm')
        print("Created 'Offshore_Norm'")
    if 'Wind_Onshore_MW' in df_processed.columns:
        df_processed['Onshore_Norm'] = (df_processed['Wind_Onshore_MW'] / df_processed['capacity_proxy']).clip(0, 1.1)
        target_cols_normalized.append('Onshore_Norm')
        print("Created 'Onshore_Norm'")

    if plot:
        plt.figure(figsize=(15, 5))
        if 'capacity_proxy' in df_processed.columns:
             plt.plot(df_processed.index, df_processed['capacity_proxy'], label='Capacity Proxy', color='green')
        if 'Wind_Offshore_MW' in df_processed.columns:
             plt.plot(df_processed.index, df_processed['Wind_Offshore_MW'], label='Offshore MW (Original)', color='blue', alpha=0.3)
        if 'Wind_Onshore_MW' in df_processed.columns:
             plt.plot(df_processed.index, df_processed['Wind_Onshore_MW'], label='Onshore MW (Original)', color='red', alpha=0.3)
        plt.title('Capacity Proxy vs Original Power')
        plt.legend()
        plt.show()

        plt.figure(figsize=(15, 5))
        if 'Offshore_Norm' in df_processed.columns:
            plt.plot(df_processed.index, df_processed['Offshore_Norm'], label='Offshore Norm', color='blue', alpha=0.7)
        if 'Onshore_Norm' in df_processed.columns:
            plt.plot(df_processed.index, df_processed['Onshore_Norm'], label='Onshore Norm', color='red', alpha=0.7)
        plt.title('Normalized Targets (Power / Capacity Proxy)')
        plt.legend()
        plt.ylim(0, 1.2)
        plt.show()

    return df_processed, target_cols_normalized

def add_lagged_features(df, normalized_target_cols, lag_hours):
    df_lagged = df.copy() # Work on a copy
    print(f"\nAdding lagged normalized targets (lag={lag_hours}H) as features...")
    lag_cols_added = []
    for col in normalized_target_cols:
        lag_col_name = f'{col}_Lag{lag_hours}H'
        df_lagged[lag_col_name] = df_lagged[col].shift(lag_hours)
        lag_cols_added.append(lag_col_name)

    print(f"NaNs introduced by shift: {df_lagged[lag_cols_added].isnull().sum().sum()}")
    return df_lagged, lag_cols_added

def select_features(df, normalized_target_cols, cols_to_remove):
    print("\n--- Separating Features and Targets ---")
    print(f"Using normalized target columns: {normalized_target_cols}")

    cols_to_exclude_from_features = [
        'Wind_Offshore_MW', 'Wind_Onshore_MW', 'Total_Wind_MW', 'capacity_proxy'
    ] + normalized_target_cols

    all_feature_cols = [col for col in df.columns if col not in cols_to_exclude_from_features]
    X_initial = df[all_feature_cols]
    y = df[normalized_target_cols]
    print(f"Initial features shape: {X_initial.shape}, Targets shape: {y.shape}")

    print("\n--- Performing Feature Selection ---")
    print(f"Removing {len(cols_to_remove)} features.")
    selected_feature_cols = [col for col in X_initial.columns if col not in cols_to_remove]
    X = X_initial[selected_feature_cols].copy()
    print(f"Selected {X.shape[1]} features. Final features shape: {X.shape}")

    return X, y

def prepare_data():
    """Loads, preprocesses, engineers features, and selects data."""
    energy_df, onshore_df, offshore_df = load_data(config.ENERGY_CSV, config.ONSHORE_CSV, config.OFFSHORE_CSV)
    merged_df = preprocess_and_merge(energy_df, onshore_df, offshore_df)
    filtered_df = filter_data(merged_df, config.START_DATE)
    proxy_df, normalized_targets = calculate_capacity_proxy_and_normalize(
        filtered_df, config.CAPACITY_PROXY_WINDOW, config.PLOT_CAPACITY_PROXY
    )
    lagged_df, lag_cols = add_lagged_features(proxy_df, normalized_targets, config.LAG_HOURS)
    X, y = select_features(lagged_df, normalized_targets, config.COLS_TO_REMOVE)

    # Return the final features, targets, and the processed df (needed for proxy later)
    return X, y, lagged_df