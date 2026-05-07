from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FINAL_DATASETS_DIR = PROJECT_ROOT / "datasets" / "final_datasets"
WIND_MODEL_DIR = PROJECT_ROOT / "wind_power_prediction"
PRICE_MODEL_DIR = PROJECT_ROOT / "energy_price_prediction"


def project_path(*parts: str) -> Path:
    """Return an absolute path under the project root."""
    return PROJECT_ROOT.joinpath(*parts)
