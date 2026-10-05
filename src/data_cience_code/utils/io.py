from pathlib import Path
import json

import pandas as pd
import matplotlib.pyplot as plt


def load_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def save_csv(path: str | Path, df: pd.DataFrame) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_json(path: str | Path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_fig(path: str | Path, fig=None, dpi: int = 150) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if fig is None:
        plt.savefig(path, bbox_inches="tight", dpi=dpi)
    else:
        fig.savefig(path, bbox_inches="tight", dpi=dpi)
