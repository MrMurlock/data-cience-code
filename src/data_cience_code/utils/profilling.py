import pandas as pd
from data_profiling import ProfileReport
from pathlib import Path


def profilling(dataset_path: str, title: str, name: str, output_dir: str | Path = "output/eda/profiling") -> str:
    """Genera un reporte de profiling HTML con data-profiling en `output_dir`."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / f"{name}.html"
    df = pd.read_csv(dataset_path)
    profile = ProfileReport(df, title=title, explorative=True)
    profile.to_file(file_path)
    return str(file_path)
