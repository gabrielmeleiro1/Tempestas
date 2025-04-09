import pandas as pd
import numpy as np
import os
import glob
import logging

#  Configuration 
WEATHER_DIR = "../datasets/wind_turbine_clusters_hourly_features"
ENERGY_DIR = "../datasets/on_and_off_shore_actual_energy_generation_2017_2025"
WEATHER_OUTPUT_DIR = os.path.join("../datasets", "processed_weather_data")
ENERGY_OUTPUT_DIR = os.path.join("../datasets", "processed_energy_data")
LOCAL_TZ = 'Europe/Amsterdam' 


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Weather Data Processing 
def process_weather_file(input_filepath, output_dir):

    base_filename = os.path.basename(input_filepath)
    output_filename = f"processed_{base_filename}"
    output_filepath = os.path.join(output_dir, output_filename)

    logging.info(f"Processing weather file: {base_filename}")
    try:
        weather_df = pd.read_csv(input_filepath)

        # Convert date column
        weather_df['date'] = pd.to_datetime(weather_df['date'])

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
        cols_to_drop = ['hour', 'day_of_week', 'month', 'day_of_year']
        for col in ['wind_direction_100m', 'wind_direction_10m']:
            if col in weather_df.columns:
                rad = np.deg2rad(weather_df[col])
                weather_df[f'{col}_sin'] = np.sin(rad)
                weather_df[f'{col}_cos'] = np.cos(rad)
                cols_to_drop.append(col)
                logging.debug(f"Encoded '{col}'.")
            else:
                logging.warning(f"Column '{col}' not found in {base_filename} for encoding.")

        # Drop original non-encoded columns
        weather_df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
        logging.debug(f"Dropped original columns for {base_filename}.")

        # Save processed file
        weather_df.to_csv(output_filepath, index=False)
        logging.info(f"Successfully processed and saved: {output_filename}")
        return True

    except Exception as e:
        logging.error(f"Failed to process weather file {base_filename}: {e}")
        return False

#  Energy Data Processing 
def process_energy_file(input_filepath, output_dir):

    base_filename = os.path.basename(input_filepath)
    output_filename = f"processed_hourly_{base_filename}"
    output_filepath = os.path.join(output_dir, output_filename)

    logging.info(f"Processing energy file: {base_filename}")
    try:
        energy_df = pd.read_csv(input_filepath)

        if 'MTU' not in energy_df.columns:
             logging.error(f"'MTU' column not found in {base_filename}. Skipping file.")
             return False

        # Parse Start Time from MTU Column
        # Handle cases where split might fail or produce unexpected results
        try:
            energy_df['start_time_str'] = energy_df['MTU'].str.split(' - ', expand=True)[0]
        except Exception as split_error:
            logging.error(f"Error splitting 'MTU' column in {base_filename}: {split_error}. Skipping file.")
            return False

        # Convert start time string to naive datetime
        naive_datetime = pd.to_datetime(energy_df['start_time_str'], format='%d.%m.%Y %H:%M', errors='coerce')

        # Drop rows where datetime parsing failed
        original_rows = len(energy_df)
        energy_df = energy_df.dropna(subset=['start_time_str']) # Drop if split failed
        naive_datetime = naive_datetime.dropna() # Drop if format conversion failed
        energy_df = energy_df.loc[naive_datetime.index] # Align dataframe with valid datetimes
        if len(energy_df) < original_rows:
            logging.warning(f"Dropped {original_rows - len(energy_df)} rows due to parsing errors in {base_filename}.")

        if energy_df.empty:
             logging.warning(f"No valid time entries found in {base_filename} after parsing. Skipping file.")
             return False

        # Localize naive datetime to local timezone (handle DST transitions)
        local_datetime = naive_datetime.dt.tz_localize(LOCAL_TZ, ambiguous='infer', nonexistent='shift_forward')

        # Convert localized datetime to UTC
        energy_df['Timestamp (UTC)'] = local_datetime.dt.tz_convert('UTC')

        # Set index and sort
        energy_df.set_index('Timestamp (UTC)', inplace=True)
        energy_df.sort_index(inplace=True)

        # Rename columns
        column_rename_map = {
            'Wind Offshore - Actual Aggregated [MW]': 'Wind_Offshore_MW',
            'Wind Onshore - Actual Aggregated [MW]': 'Wind_Onshore_MW',
            'Solar - Actual Aggregated [MW]': 'Solar_MW'
        }
        # Only rename columns that actually exist in the file
        existing_cols_to_rename = {k: v for k, v in column_rename_map.items() if k in energy_df.columns}
        energy_df.rename(columns=existing_cols_to_rename, inplace=True)
        logging.debug(f"Renamed columns for {base_filename}: {existing_cols_to_rename}")

        # Define target columns based on renamed existing columns
        target_cols = [col for col in ['Wind_Offshore_MW', 'Wind_Onshore_MW', 'Solar_MW'] if col in energy_df.columns]

        if not target_cols:
             logging.warning(f"No target columns (Wind/Solar MW) found in {base_filename} after renaming. Skipping file.")
             return False

        # Select relevant columns
        energy_targets_df = energy_df[target_cols].copy()

        # Convert potential object/string types to numeric, coercing errors
        for col in target_cols:
            energy_targets_df[col] = pd.to_numeric(energy_targets_df[col], errors='coerce')

        # Resample to hourly frequency, calculating the mean
        # Use numeric_only=True if pandas version requires it and non-numeric columns exist (shouldn't after selection/coercion)
        try:
             energy_targets_hourly_df = energy_targets_df.resample('h').mean() 
        except TypeError as resample_err:
             logging.error(f"Resampling error for {base_filename} (maybe non-numeric data persists?): {resample_err}")
             energy_targets_hourly_df = energy_targets_df.resample('h').agg(lambda x: pd.to_numeric(x, errors='coerce').mean())


        # Check for missing values AFTER resampling and numeric conversion
        missing_before_interp = energy_targets_hourly_df.isnull().sum()
        if missing_before_interp.sum() > 0:
            logging.warning(f"Missing values found after resampling in {base_filename}:\n{missing_before_interp}")
            energy_targets_hourly_df.interpolate(method='time', inplace=True)
            logging.info(f"Interpolated missing values in {base_filename}")

        # Drop Solar_MW (not used in our model, but could for future use)
        if 'Solar_MW' in energy_targets_hourly_df.columns:
            energy_targets_hourly_df.drop('Solar_MW', axis=1, inplace=True)
            logging.debug(f"Dropped Solar_MW column for {base_filename}")

        # Save processed file
        energy_targets_hourly_df.to_csv(output_filepath, index=True) # Index is Timestamp (UTC)
        logging.info(f"Successfully processed and saved: {output_filename}")
        return True

    except Exception as e:
        logging.error(f"Failed to process energy file {base_filename}: {e}", exc_info=True) # Log traceback
        return False

#  Main Execution 
if __name__ == "__main__":
    logging.info("Starting script...")

    #  Process Weather Files 
    logging.info(f"Looking for weather files in: {WEATHER_DIR}")
    weather_files = glob.glob(os.path.join(WEATHER_DIR, "*.csv"))

    if not weather_files:
        logging.warning(f"No CSV files found in {WEATHER_DIR}")
    else:
        logging.info(f"Found {len(weather_files)} weather files to process.")
        # Create output directory if it doesn't exist
        os.makedirs(WEATHER_OUTPUT_DIR, exist_ok=True)
        logging.info(f"Weather output directory: {WEATHER_OUTPUT_DIR}")

        success_count = 0
        fail_count = 0
        for filepath in weather_files:
            if process_weather_file(filepath, WEATHER_OUTPUT_DIR):
                success_count += 1
            else:
                fail_count += 1
        logging.info(f"Weather file processing complete. Success: {success_count}, Failed: {fail_count}")

    #  Process Energy Files 
    logging.info(f"\nLooking for energy files in: {ENERGY_DIR}")
    energy_files = glob.glob(os.path.join(ENERGY_DIR, "*.csv"))

    if not energy_files:
        logging.warning(f"No CSV files found in {ENERGY_DIR}")
    else:
        logging.info(f"Found {len(energy_files)} energy files to process.")
        # Create output directory if it doesn't exist
        os.makedirs(ENERGY_OUTPUT_DIR, exist_ok=True)
        logging.info(f"Energy output directory: {ENERGY_OUTPUT_DIR}")

        success_count = 0
        fail_count = 0
        for filepath in energy_files:
            # Skip processing files inside the newly created output directory
            if os.path.dirname(filepath) == ENERGY_OUTPUT_DIR:
                continue
            if process_energy_file(filepath, ENERGY_OUTPUT_DIR):
                success_count += 1
            else:
                fail_count += 1
        logging.info(f"Energy file processing complete. Success: {success_count}, Failed: {fail_count}")

    logging.info("\nScript finished.")