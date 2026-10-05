import json
from pathlib import Path
from typing import Any, Union

import pandas as pd
import matplotlib.pyplot as plt


class ArtifactsStore:
    def __init__(
        self,
        tables_dir: Union[str, Path],
        metrics_dir: Union[str, Path],
        images_dir: Union[str, Path],
    ) -> None:
        self.tables_dir = Path(tables_dir)
        self.metrics_dir = Path(metrics_dir)
        self.images_dir = Path(images_dir)
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)

    def save_table(self, name: str, df: pd.DataFrame) -> Path:
        path = self.tables_dir / f"{name}.csv"
        df.to_csv(path, index=False)
        return path

    def save_metric(self, name: str, obj: Any) -> Path:
        path = self.metrics_dir / f"{name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        return path

    def save_image(self, name: str, fig: Union[plt.Figure, Any] = None, dpi: int = 150) -> Path:
        path = self.images_dir / f"{name}.png"
        if fig is None:
            plt.savefig(path, bbox_inches="tight", dpi=dpi)
        else:
            fig.savefig(path, bbox_inches="tight", dpi=dpi)
        return path
