import pandas as pd
import numpy as np
import os
import glob
import logging

# --- Configuration ---
# Directory where the *processed* weather files are located
PROCESSED_WEATHER_DIR = "datasets/processed_weather_data"

# Output file names
ONSHORE_OUTPUT_FILE = os.path.join(PROCESSED_WEATHER_DIR, "combined_onshore_weather.csv")
OFFSHORE_OUTPUT_FILE = os.path.join(PROCESSED_WEATHER_DIR, "combined_offshore_weather.csv")

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Function to Combine and Average Files ---
def combine_and_average_weather(file_pattern, output_filepath):
    """
    Finds files matching a pattern, reads them, combines them,
    averages features by timestamp, and saves the result.
    """
    logging.info(f"Looking for files matching pattern: {file_pattern}")
    file_paths = glob.glob(file_pattern)

    if not file_paths:
        logging.warning(f"No files found matching pattern: {file_pattern}")
        return False

    logging.info(f"Found {len(file_paths)} files to combine and average.")

    all_dataframes = []
    for filepath in file_paths:
        try:
            # Read the CSV
            df = pd.read_csv(filepath)

            # Ensure 'date' column exists
            if 'date' not in df.columns:
                logging.warning(f"'date' column not found in {os.path.basename(filepath)}. Skipping this file.")
                continue

            # Convert 'date' to datetime and set as index *IMPORTANT*
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)

            all_dataframes.append(df)
            logging.debug(f"Successfully read and prepared: {os.path.basename(filepath)}")

        except Exception as e:
            logging.error(f"Failed to read or process file {os.path.basename(filepath)}: {e}")

    if not all_dataframes:
        logging.error(f"No valid DataFrames could be loaded for pattern: {file_pattern}. Cannot create output.")
        return False

    # Concatenate all dataframes. Rows with the same index (date) will be stacked.
    logging.info("Concatenating DataFrames...")
    combined_df = pd.concat(all_dataframes)

    # Group by the index (date) and calculate the mean for each timestamp.
    # This averages the features across all clusters for that specific hour.
    # numeric_only=True ensures mean is only calculated for numeric columns
    logging.info("Averaging features by timestamp...")
    try:
        averaged_df = combined_df.groupby(combined_df.index).mean(numeric_only=True)
    except TypeError:
         # Fallback for older pandas versions potentially
         logging.warning("groupby().mean() encountered TypeError, attempting manual numeric selection.")
         numeric_cols = combined_df.select_dtypes(include=np.number).columns
         averaged_df = combined_df.groupby(combined_df.index)[numeric_cols].mean()


    # Sort by time index
    averaged_df.sort_index(inplace=True)

    # Save the averaged data
    logging.info(f"Saving averaged data to: {output_filepath}")
    averaged_df.to_csv(output_filepath, index=True) # index=True saves the 'date' index
    logging.info(f"Successfully saved {output_filepath}")
    return True

# --- Main Execution ---
if __name__ == "__main__":
    logging.info("Starting weather data combination script...")

    # --- Combine Onshore Files ---
    onshore_pattern = os.path.join(PROCESSED_WEATHER_DIR, "processed_on_*.csv")
    combine_and_average_weather(onshore_pattern, ONSHORE_OUTPUT_FILE)

    # --- Combine Offshore Files ---
    offshore_pattern = os.path.join(PROCESSED_WEATHER_DIR, "processed_off_*.csv")
    combine_and_average_weather(offshore_pattern, OFFSHORE_OUTPUT_FILE)

    logging.info("Weather data combination script finished.")