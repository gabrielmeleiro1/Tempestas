from wind_power_prediction.feature_schema import price_feature_columns, tcn_feature_columns


def test_tcn_feature_columns_match_trained_model_contract():
    feature_cols = tcn_feature_columns(lag_hours=1)

    assert len(feature_cols) == 21
    assert len(feature_cols) == len(set(feature_cols))
    assert feature_cols[-2:] == ["Offshore_Norm_Lag1H", "Onshore_Norm_Lag1H"]
    assert "wind_speed_100m_offshore" in feature_cols
    assert "wind_direction_100m_cos_onshore" in feature_cols


def test_price_feature_columns_include_wind_time_and_lags():
    feature_cols = price_feature_columns(price_lags=[1, 6, 24])

    assert feature_cols == [
        "Predicted_Offshore_MW",
        "Predicted_Onshore_MW",
        "hour",
        "dayofweek",
        "dayofyear",
        "month",
        "year",
        "weekofyear",
        "price_lag_1h",
        "price_lag_6h",
        "price_lag_24h",
    ]
