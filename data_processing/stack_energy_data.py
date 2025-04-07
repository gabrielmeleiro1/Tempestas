import pandas as pd
import os
import glob
import logging

# --- Configuration ---
# Directory where the *processed* energy files are located
PROCESSED_ENERGY_DIR = "datasets/processed_energy_data" # Corrected based on your example path structure

# Output file name for the final combined data
COMBINED_ENERGY_OUTPUT_FILE = os.path.join(PROCESSED_ENERGY_DIR, "combined_total_energy_data_2017_2025.csv")

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Main Execution ---
if __name__ == "__main__":
    logging.info("Starting energy data combination script...")

    # Define the pattern to find the processed hourly energy files
    energy_file_pattern = os.path.join(PROCESSED_ENERGY_DIR, "processed_hourly_*.csv")
    logging.info(f"Looking for files matching pattern: {energy_file_pattern}")

    file_paths = glob.glob(energy_file_pattern)

    if not file_paths:
        logging.warning(f"No files found matching pattern: {energy_file_pattern}. Exiting.")
        exit() # Exit if no files are found

    logging.info(f"Found {len(file_paths)} energy files to combine.")

    all_energy_dataframes = []
    for filepath in file_paths:
        filename = os.path.basename(filepath)
        # Skip the potential combined file if the script is run multiple times
        if filename == os.path.basename(COMBINED_ENERGY_OUTPUT_FILE):
            logging.debug(f"Skipping already combined file: {filename}")
            continue
        try:
            # Read the CSV, ensuring the timestamp is the index and parsed correctly
            # The previous script saved the index, so we read it back using index_col
            df = pd.read_csv(filepath, index_col='Timestamp (UTC)', parse_dates=True)
            all_energy_dataframes.append(df)
            logging.debug(f"Successfully read: {filename}")

        except Exception as e:
            logging.error(f"Failed to read or process file {filename}: {e}")

    if not all_energy_dataframes:
        logging.error("No valid energy DataFrames could be loaded. Cannot create output.")
        exit() # Exit if no dataframes were loaded

    # Concatenate all dataframes vertically (end-to-end)
    logging.info("Concatenating DataFrames...")
    combined_df = pd.concat(all_energy_dataframes, axis=0) # axis=0 is default, stacks rows

    # Sort the combined dataframe by the timestamp index to ensure chronological order
    logging.info("Sorting combined data by timestamp...")
    combined_df.sort_index(inplace=True)

    # Check for and handle potential duplicate timestamps (e.g., from file overlaps)
    duplicates = combined_df.index.duplicated().sum()
    if duplicates > 0:
        logging.warning(f"Found {duplicates} duplicate timestamps (indices). Keeping the first occurrence.")
        combined_df = combined_df[~combined_df.index.duplicated(keep='first')]

    # Save the final combined data
    logging.info(f"Saving combined energy data to: {COMBINED_ENERGY_OUTPUT_FILE}")
    combined_df.to_csv(COMBINED_ENERGY_OUTPUT_FILE, index=True) # index=True saves the 'Timestamp (UTC)' index
    logging.info(f"Successfully saved {COMBINED_ENERGY_OUTPUT_FILE}")

    logging.info("Energy data combination script finished.")