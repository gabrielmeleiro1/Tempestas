from pathlib import Path

from wind_power_prediction import config
from wind_power_prediction.paths import FINAL_DATASETS_DIR, PROJECT_ROOT


def test_project_root_is_repo_root():
    assert PROJECT_ROOT == Path(__file__).resolve().parents[1]
    assert (PROJECT_ROOT / "README.md").exists()


def test_training_config_uses_repo_root_data_paths():
    assert config.BASE_DIR == PROJECT_ROOT
    assert config.ENERGY_CSV == FINAL_DATASETS_DIR / "combined_total_energy_data_2017_2025.csv"
    assert config.ONSHORE_CSV == FINAL_DATASETS_DIR / "final_averaged_onshore_weather.csv"
    assert config.OFFSHORE_CSV == FINAL_DATASETS_DIR / "final_averaged_offshore_weather.csv"
