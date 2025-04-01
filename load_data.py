import logging
import os
import time
from typing import List, Dict, Any

import numpy as np
import openmeteo_requests
import pandas as pd
import requests # Import requests for base session and HTTPError handling
import requests_cache
# Import the necessary components for manual retry configuration (not working automatically)
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry



# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define Dutch locations (Cities and Offshore Wind Sites)
# Using \u00A0 (non-breaking space) if needed, otherwise standard spaces are fine.
# Assuming standard spaces from your input.
DUTCH_LOCATIONS: List[Dict[str, Any]] = [
    {"name": "Flevoland Windplan Groen", "lat": 52.56, "lon": 5.76},
    {"name": "Windpark Noordoostpolder", "lat": 52.721, "lon": 5.583},
    {"name": "Windplan Blauw", "lat": 52.56984, "lon": 5.650891},
    {"name": "Windpark Zeewolde Princess", "lat": 52.85647, "lon": 4.93378},
    {"name": "Alexia Windpark", "lat": 52.8222, "lon": 4.932895},
    {"name": "Groningen Westereems", "lat": 53.4474, "lon": 6.842},
    {"name": "Delfzijl-Noord", "lat": 53.3215, "lon": 6.9701},
    {"name": "Windpark Oostpolder Geefsweer", "lat": 53.321651, "lon": 6.969826},
    {"name": "Borssele I & II", "lat": 51.70278, "lon": 3.07611},
    {"name": "Gemini Wind Farm", "lat": 54.03611, "lon": 5.96306},
    {"name": "Hollandse Kust (Zuid) I & II", "lat": 52.36667, "lon": 4.11667},
    {"name": "Hollandse Kust (Zuid) III & IV Hollandse Kust Noord", "lat": 52.71511, "lon": 4.251},
    {"name": "Eneco Luchterduinen", "lat": 52.40481, "lon": 4.161821},
    {"name": "IJsselmeer", "lat": 52.71505, "lon": 5.578903},
    {"name": "Fryslan Wind Farm", "lat": 52.99435, "lon": 5.267503}
]

# Define desired hourly weather variables for Wind and Solar forecasting
HOURLY_VARIABLES: List[str] = [
    # Wind Variables
    "wind_speed_100m",
    "wind_direction_100m",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "surface_pressure",
    "temperature_2m",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    "relative_humidity_2m",
    "rain",
    # "shortwave_radiation",
    # "direct_normal_irradiance",
    # "cloud_cover",
    # "dew_point_2m",
    # "diffuse_radiation",
]


#  today's date at midnight UTC for consistent calculations
today = pd.Timestamp('today', tz='UTC').normalize()
# Go back 5 years for the start date
start_date_str = (today - pd.Timedelta(days=365 * 5)).strftime('%Y-%m-%d')
# End date is 5 days ago - ERA5/Best Match usually has a ~5 day delay
end_date_str = (today - pd.Timedelta(days=5)).strftime('%Y-%m-%d')

# Output Directory
OUTPUT_DIR = 'datasets'
RATE_LIMIT_WAIT_SECONDS = 60 # Wait time in seconds when rate limit is hit
MAX_RATE_LIMIT_RETRIES = 5   # Max attempts for a single location after hitting rate limits


def setup_openmeteo_client() -> openmeteo_requests.Client:
    """Sets up the Open-Meteo client with caching and manual retries."""
    # Create a standard requests Session
    base_session = requests.Session()

    # --- Manually configure the retry strategy ---
    retry_strategy = Retry(
        total=3,                 # Total number of retries for all errors
        backoff_factor=1,        # Wait 1s, 2s, 4s between retries
        status_forcelist=[500, 502, 503, 504], # Status codes to retry on

    )

    # Create an HTTP adapter with this retry strategy
    adapter = HTTPAdapter(max_retries=retry_strategy)

    # Mount it for both http and https protocols
    base_session.mount("https://", adapter)
    base_session.mount("http://", adapter)

    # Cache API responses using the session that now has the retry adapter
    # expire_after=-1 means cache never expires
    cached_retry_session = requests_cache.CachedSession('.cache', expire_after=-1, session=base_session) # Use the configured base_session

    logger.info("Open-Meteo client setup complete with caching and manual retries.")
    return openmeteo_requests.Client(session=cached_retry_session)


def make_api_request(
    openmeteo: openmeteo_requests.Client,
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    hourly_variables: List[str]
) -> List[Any]: # Returns list of response objects
    """Makes the API request to Open-Meteo."""
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": hourly_variables,
        "timezone": "UTC" # Explicitly request UTC time
    }
    logger.debug(f"Requesting data for {latitude},{longitude} from {start_date} to {end_date}")
    # The session configured in setup_openmeteo_client will handle 5xx retries automatically
    responses = openmeteo.weather_api(url, params=params)
    logger.debug(f"API response received for {latitude},{longitude}")
    return responses
    # Note: The session might still raise exceptions for non-retryable errors (e.g., 400 Bad Request)
    # or after exhausting retries for 5xx errors, which will be caught in process_location_data.

def process_api_response(response: Any, requested_variables: List[str]) -> pd.DataFrame:
    """Processes the API response into a Pandas DataFrame."""
    hourly = response.Hourly()
    logger.debug(f"Processing response for location {response.Latitude()}, {response.Longitude()}")

    # Create the time range index first
    try:
        time_index = pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        )
    except Exception as e:
        logger.error(f"Error creating date range: {e}. Time: {hourly.Time()}, End: {hourly.TimeEnd()}, Interval: {hourly.Interval()}")
        # Return an empty DataFrame or raise error depending on desired handling
        return pd.DataFrame()

    hourly_data = {"date": time_index}

    # Dynamically map variables based on the requested order
    for i, var_name in enumerate(requested_variables):
        # Check if the variable index exists in the response
        # Access Variables using the index i, corresponding to the order in HOURLY_VARIABLES
        variable_obj = hourly.Variables(i)
        if variable_obj is not None: # Check if variable object exists at index i
            try:
                values = variable_obj.ValuesAsNumpy()
                 # Basic check for data consistency
                if len(values) == len(time_index):
                    hourly_data[var_name] = values
                    logger.debug(f"Successfully processed variable: {var_name}")
                else:
                     logger.warning(f"Length mismatch for variable '{var_name}' ({len(values)}) and time index ({len(time_index)}) at {response.Latitude()}, {response.Longitude()}. Filling with NaNs.")
                     hourly_data[var_name] = np.full(len(time_index), np.nan)

            except Exception as e:
                logger.error(f"Error processing variable '{var_name}' (index {i}) at {response.Latitude()}, {response.Longitude()}: {e}. Filling with NaNs.")
                hourly_data[var_name] = np.full(len(time_index), np.nan)
        else:
             logger.warning(f"Variable '{var_name}' (expected index {i}) not found or is null in API response for {response.Latitude()}, {response.Longitude()}. Filling with NaNs.")
             hourly_data[var_name] = np.full(len(time_index), np.nan)

    return pd.DataFrame(data=hourly_data)

def sanitize_filename(name: str) -> str:
    """Removes or replaces characters unsuitable for filenames."""
    name = name.lower()
    # Replace spaces and common separators with underscores
    for char in [' ', '(', ')', '/', '\\', "'", '&']: # Added '&'
        name = name.replace(char, '_')
    name = '_'.join(filter(None, name.split('_')))
    # Handle potential leading/trailing underscores
    name = name.strip('_')
    return name


def save_to_csv(dataframe: pd.DataFrame, csv_filename: str):
    """Saves the DataFrame to a CSV file."""
    try:
        # Ensure the directory exists
        # Handle potential errors if dirname is empty or invalid
        dir_name = os.path.dirname(csv_filename)
        if dir_name: # Only create if dirname is not empty
             os.makedirs(dir_name, exist_ok=True)
        dataframe.to_csv(csv_filename, index=False, date_format='%Y-%m-%dT%H:%M:%S')
        logger.info(f"Data successfully saved to {csv_filename}")
    except Exception as e:
        logger.error(f"Failed to save data to {csv_filename}: {e}")

def process_location_data(
    openmeteo: openmeteo_requests.Client,
    location_info: Dict[str, Any],
    start_date: str,
    end_date: str,
    hourly_variables: List[str]
):
    """Fetches, processes, and saves weather data for a single location,
       handling rate limits with retries."""
    latitude = location_info['lat']
    longitude = location_info['lon']
    location_name = location_info['name']
    safe_name = sanitize_filename(location_name)

    # Ensure OUTPUT_DIR exists before joining path
    if not os.path.exists(OUTPUT_DIR):
         try:
             os.makedirs(OUTPUT_DIR)
             logger.info(f"Created output directory: {OUTPUT_DIR}")
         except OSError as e:
             logger.error(f"Could not create output directory {OUTPUT_DIR}: {e}")
             return # Cannot proceed without output directory

    csv_filename = os.path.join(OUTPUT_DIR, f'{safe_name}_weather.csv')

    # Skip if file already exists
    if os.path.exists(csv_filename):
        logger.info(f"Weather data for '{location_name}' ({safe_name}) already exists. Skipping...")
        return

    logger.info(f"--- Processing location: {location_name} ({latitude}, {longitude}) ---")

    retries = 0
    while retries <= MAX_RATE_LIMIT_RETRIES:
        try:
            # --- Attempt API Call and Processing ---
            # Make API request (returns a list, but we expect one response for one location)
            responses = make_api_request(openmeteo, latitude, longitude, start_date, end_date, hourly_variables)

            if not responses:
                logger.error(f"No response received from API for {location_name}.")
                # Decide if this is retryable - assuming not for now
                return # Exit processing for this location

            response = responses[0] # Process the first response

            # Process the response into a DataFrame
            dataframe = process_api_response(response, hourly_variables)

            if dataframe.empty:
                logger.warning(f"Processing resulted in an empty DataFrame for {location_name}. Skipping save.")
                return # Exit processing for this location

            # Save the DataFrame to CSV
            save_to_csv(dataframe, csv_filename)

            logger.info(f"Successfully processed and saved data for {location_name}")
            break # <<< Exit the while loop on success

        except requests.exceptions.HTTPError as http_err:
            # Handles HTTP Errors (especially Rate Limits)
            # Check if the error has a response attribute before accessing status_code
            if http_err.response is not None and http_err.response.status_code == 429:
                retries += 1
                if retries <= MAX_RATE_LIMIT_RETRIES:
                    logger.warning(
                        f"Rate limit hit (429) for {location_name}. "
                        f"Waiting {RATE_LIMIT_WAIT_SECONDS} seconds... "
                        f"(Attempt {retries}/{MAX_RATE_LIMIT_RETRIES})"
                    )
                    time.sleep(RATE_LIMIT_WAIT_SECONDS)
                else:
                    logger.error(
                        f"Max rate limit retries ({MAX_RATE_LIMIT_RETRIES}) reached "
                        f"for {location_name}. Skipping this location."
                    )
                    return # Stop processing this location entirely
            else:
                # Handle other HTTP errors (e.g., 400 Bad Request, 404 Not Found)
                # 5xx errors should have been handled by the retry adapter already,
                # but might reach here if retries are exhausted.
                logger.error(f"HTTP error processing data for {location_name}: {http_err}")
                # Stop processing this specific location on other HTTP errors.
                return # Exit processing for this location

        except Exception as e:
            #  Handle other potential errors 
            # (e.g., network issues not caught by adapter, errors in process_api_response, save_to_csv)
            logger.error(f"An unexpected error occurred processing data for {location_name}: {e}", exc_info=True) # Log traceback
            # Stop processing this specific location on unexpected errors
            return # Exit processing for this location

# --- Main Execution ---

if __name__ == '__main__':
    # Ensure logging is configured 
    if not logger.hasHandlers():
         logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    logger.info("Starting weather data fetching process...")
    logger.info(f"Fetching data from {start_date_str} to {end_date_str}")
    logger.info(f"Hourly variables requested: {', '.join(HOURLY_VARIABLES)}")

    # Create output directory if it doesn't exist (redundant check, but safe)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Setup Open-Meteo client 
    openmeteo_client = setup_openmeteo_client()

    # Iterate through locations and process data
    total_locations = len(DUTCH_LOCATIONS)
    for i, location in enumerate(DUTCH_LOCATIONS):
        logger.info(f"--- Progress: {i+1}/{total_locations} ---")
        process_location_data(
            openmeteo_client,
            location,
            start_date_str,
            end_date_str,
            HOURLY_VARIABLES
        )
        # Add a small delay *regardless* of success/failure/rate limit for a lot of requests
        # This can help prevent hitting the rate limit in the first place,
        # especially if MAX_RATE_LIMIT_RETRIES is low.
        # time.sleep(0.2) # e.g., wait 0.2 seconds between locations

    logger.info("Weather data fetching process finished.")