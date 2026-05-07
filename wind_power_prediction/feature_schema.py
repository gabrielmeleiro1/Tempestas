from collections.abc import Iterable


BASE_TCN_FEATURE_COLUMNS = [
    "wind_speed_100m_offshore",
    "surface_pressure_offshore",
    "temperature_2m_offshore",
    "relative_humidity_2m_offshore",
    "rain_offshore",
    "hour_sin_offshore",
    "hour_cos_offshore",
    "day_of_week_sin_offshore",
    "day_of_week_cos_offshore",
    "day_of_year_sin_offshore",
    "day_of_year_cos_offshore",
    "wind_direction_100m_sin_offshore",
    "wind_direction_100m_cos_offshore",
    "wind_speed_100m_onshore",
    "temperature_2m_onshore",
    "relative_humidity_2m_onshore",
    "rain_onshore",
    "wind_direction_100m_sin_onshore",
    "wind_direction_100m_cos_onshore",
]

PRICE_BASE_FEATURE_COLUMNS = [
    "Predicted_Offshore_MW",
    "Predicted_Onshore_MW",
    "hour",
    "dayofweek",
    "dayofyear",
    "month",
    "year",
    "weekofyear",
]


def tcn_feature_columns(lag_hours: int = 1) -> list[str]:
    return [
        *BASE_TCN_FEATURE_COLUMNS,
        f"Offshore_Norm_Lag{lag_hours}H",
        f"Onshore_Norm_Lag{lag_hours}H",
    ]


def price_feature_columns(price_lags: Iterable[int]) -> list[str]:
    return [
        *PRICE_BASE_FEATURE_COLUMNS,
        *[f"price_lag_{lag}h" for lag in price_lags],
    ]
