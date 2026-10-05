from pathlib import Path

import matplotlib.pyplot as plt

from data_cience_code.utils.artifacts import ArtifactsStore


ARTIFACTS_DIR = Path("artifacts/eda")
TABLES_DIR = ARTIFACTS_DIR / "tables"
METRICS_DIR = ARTIFACTS_DIR / "metrics"
IMAGES_DIR = ARTIFACTS_DIR / "images"

REPORTS_EDA_DIR = Path("reports/eda")
OUTPUTS_EDA_DIR = Path("output/eda")
PROFILING_EDA_DIR = OUTPUTS_EDA_DIR / "profiling"


def ensure_dirs() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_EDA_DIR.mkdir(parents=True, exist_ok=True)


artifacts = ArtifactsStore(
    tables_dir=TABLES_DIR,
    metrics_dir=METRICS_DIR,
    images_dir=IMAGES_DIR,
)


def save_image(name: str, fig) -> Path:
    """Guarda la figura como artifact y cierra la figura de matplotlib."""
    path = artifacts.save_image(name, fig)
    plt.close(fig)
    return path
