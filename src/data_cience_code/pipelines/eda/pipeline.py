"""Punto de composición del pipeline EDA (docs/1.EDA.md §10).

`main()` compone las etapas del análisis; cada responsabilidad vive en un
módulo propio:

1. profiling inicial y post-limpieza (`utils.profilling`).
2. preprocessing del dataset (`preprocessing`).
3. calidad de datos (`quality`).
4. análisis univariado (`univariate`).
5. análisis bivariado (`bivariate`).
6. render del reporte y la presentación (`render_eda`).

El texto narrativo de los documentos vive en los templates Markdown de
`reports/eda/`; Python solo produce datos y artifacts.
"""
import pandas as pd

from data_cience_code.config.const import CLEAN_DATASET_PATH
from data_cience_code.config.const import RAW_DATASET_PATH
from data_cience_code.utils.profilling import profilling

from .bivariate import save_bivariate_artifacts
from .config_eda import PROFILING_EDA_DIR
from .config_eda import artifacts
from .config_eda import ensure_dirs
from .preprocessing import preprocess_dataset
from .quality import (
    categorical_cardinality,
    dataset_overview,
    duplicate_stats,
    missing_raw_summary,
    missing_values,
    numeric_ranges,
)
from .render_eda import export_pptx
from .render_eda import render_reports
from .univariate import save_univariate_artifacts


def run_analysis() -> None:
    """Prepara el dataset, ejecuta el análisis y genera todos los artifacts."""
    ensure_dirs()

    # 1. Profiling del dataset original
    df_raw = pd.read_csv(RAW_DATASET_PATH)
    profilling(
        RAW_DATASET_PATH,
        "Original Dataset Profiling Report",
        "original_profiling",
        output_dir=PROFILING_EDA_DIR,
    )

    # 2. Calidad sobre datos crudos: faltantes previos a la imputación
    artifacts.save_table("missing_values_raw", missing_values(df_raw))
    artifacts.save_metric("missing_raw_summary", missing_raw_summary(df_raw))

    # 3. Preprocessing y guardado del dataset limpio
    df = preprocess_dataset(df_raw)
    df.to_csv(CLEAN_DATASET_PATH, index=False)
    profilling(
        CLEAN_DATASET_PATH,
        "Cleaned Dataset Profiling Report",
        "clean_profiling",
        output_dir=PROFILING_EDA_DIR,
    )

    # 4. Calidad de datos (post-preprocessing)
    artifacts.save_table("dataset_overview", dataset_overview(df))
    artifacts.save_table("missing_values", missing_values(df))
    artifacts.save_metric("duplicate_stats", duplicate_stats(df))
    artifacts.save_table("categorical_cardinality", categorical_cardinality(df))
    artifacts.save_table("numeric_ranges", numeric_ranges(df))

    # 5. Análisis univariado y bivariado -> artifacts
    save_univariate_artifacts(df)
    save_bivariate_artifacts(df)


def main(render_documents: bool = True, pptx: bool = False) -> None:
    """Ejecuta el pipeline: análisis y, por defecto, render de los documentos."""
    run_analysis()
    if render_documents:
        render_reports()
        if pptx:
            export_pptx()
