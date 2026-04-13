import pandas as pd
from config import config


def load_csv(filename: str, folder: str = "raw") -> pd.DataFrame:
    """Load CSV file from dataset folder (raw or processed)."""
    path = config.get_dataset_path(folder) + f"/{filename}"
    # keep_default_na=False evita que "N/A" se interprete como NaN
    return pd.read_csv(path, keep_default_na=False, na_values=[""])


def load_dataset(filename: str = "original.csv", folder: str = "raw") -> pd.DataFrame:
    """Load dataset with default filename."""
    return load_csv(filename, folder)