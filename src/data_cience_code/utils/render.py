#!/usr/bin/env python3
import json
import os
import re
from pathlib import Path
from typing import Any, Optional

import pandas as pd


def fmt_num(x: float | int) -> str:
    if isinstance(x, float):
        if x != 0 and abs(x) < 0.01:
            # valores muy pequeños (ej. p-values) en notación científica
            return f"{x:.2e}"
        return f"{x:.2f}"
    return str(x)


def format_metric(val: Any) -> str:
    if isinstance(val, dict):
        if len(val) == 0:
            return "{}"
        return ", ".join(
            f"{k}: {fmt_num(v) if isinstance(v, (int, float)) else str(v)}"
            for k, v in val.items()
        )
    if isinstance(val, list):
        if len(val) == 0:
            return "[]"
        return ", ".join(fmt_num(x) if isinstance(x, (int, float)) else str(x) for x in val)
    return fmt_num(val) if isinstance(val, (int, float)) else str(val)


def df_to_md(df: pd.DataFrame, offset: int = 0, limit: Optional[int] = None) -> str:
    if df is None or df.empty:
        return "*N/A*"
    df_fmt = df.copy()
    for c in df_fmt.select_dtypes(include=["float64", "float32"]).columns:
        df_fmt[c] = df_fmt[c].apply(fmt_num)

    if offset >= len(df_fmt):
        return "*N/A*"
    end = len(df_fmt) if limit is None else offset + limit
    if end > len(df_fmt):
        end = len(df_fmt)
    df_fmt = df_fmt.iloc[offset:end]

    if df_fmt.empty:
        return "*N/A*"

    return df_fmt.to_markdown(index=False)


class Renderer:
    def __init__(
        self,
        artifacts_base: Path,
        tables_dir: Optional[Path] = None,
        metrics_dir: Optional[Path] = None,
        images_dir: Optional[Path] = None,
    ) -> None:
        self.artifacts_base = Path(artifacts_base)
        self.tables_dir = Path(tables_dir) if tables_dir else self.artifacts_base / "tables"
        self.metrics_dir = Path(metrics_dir) if metrics_dir else self.artifacts_base / "metrics"
        self.images_dir = Path(images_dir) if images_dir else self.artifacts_base / "images"
        # Registro de artifacts no resueltados durante el último render_file().
        self.last_missing: list[str] = []

    def load_table(self, name: str) -> Optional[pd.DataFrame]:
        p = self.tables_dir / f"{name}.csv"
        if p.exists():
            try:
                return pd.read_csv(p)
            except Exception:
                return None
        return None

    def load_metric(self, name: str) -> Any:
        p = self.metrics_dir / f"{name}.json"
        if p.exists():
            try:
                with open(p, encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    def _rel_path_from_md(self, target: Path, md_path: Path) -> str:
        try:
            rel = Path(os.path.relpath(target.resolve(), md_path.parent.resolve()))
            return rel.as_posix()
        except Exception:
            return target.as_posix()

    def render_file(self, template_path: Path, output_path: Path) -> Path:
        template_path = Path(template_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        text = template_path.read_text(encoding="utf-8")

        missing: list[str] = []

        def repl_table(m):
            name = m.group(1)
            df = self.load_table(name)
            if df is None:
                missing.append(f"table:{name}")
                return "*missing table*"
            if df.empty:
                return "*N/A*"
            return df_to_md(df)

        def repl_table_chunk(m):
            name = m.group(1)
            offset_str = m.group(2)
            limit_str = m.group(3) if len(m.groups()) >= 3 and m.group(3) is not None else None
            df = self.load_table(name)
            if df is None:
                missing.append(f"table:{name}")
                return "*missing table*"
            if df.empty:
                return "*N/A*"
            try:
                offset = int(offset_str) if offset_str is not None else 0
            except Exception:
                offset = 0
            try:
                limit = int(limit_str) if limit_str is not None else None
            except Exception:
                limit = None
            return df_to_md(df, offset=offset, limit=limit)

        def repl_metric(m):
            name = m.group(1)
            val = self.load_metric(name)
            if val is None:
                missing.append(f"metric:{name}")
                return f"*missing metric: {name}*"
            return format_metric(val)

        def repl_image(m):
            name = m.group(1)
            width = m.group(2)
            img_path = self.images_dir / f"{name}.png"
            if not img_path.exists():
                missing.append(f"image:{name}")
                return f"*missing image: {name}*"
            rel = self._rel_path_from_md(img_path, output_path)
            if width:
                # alt "w:<px>" es la sintaxis de Marp para fijar el ancho de la imagen
                return f"![w:{width}]({rel})"
            return f"![{name}]({rel})"

        text = re.sub(r'\{\{\s*table\("([^"]+)"\s*,\s*(\d+)\s*,\s*(\d+)\s*\)\s*\}\}', repl_table_chunk, text)
        text = re.sub(r'\{\{\s*table\("([^"]+)"\s*,\s*(\d+)\s*\)\s*\}\}', repl_table_chunk, text)
        text = re.sub(r'\{\{\s*table\("([^"]+)"\)\s*\}\}', repl_table, text)
        text = re.sub(r'\{\{\s*metric\("([^"]+)"\)\s*\}\}', repl_metric, text)
        text = re.sub(r'\{\{\s*image\("([^"]+)"(?:,\s*(\d+))?\s*\)\s*\}\}', repl_image, text)

        self.last_missing = missing

        output_path.write_text(text, encoding="utf-8")
        return output_path
