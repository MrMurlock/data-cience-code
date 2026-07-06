import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import data_cience_code.config.config as config


def normalize_numerical_features(
    df: pd.DataFrame, numerical_columns: list[str]
) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Normaliza variables numéricas usando StandardScaler.

    Args:
        df: DataFrame con los datos a normalizar
        numerical_columns: Lista de nombres de columnas numéricas

    Returns:
        Tuple con:
        - DataFrame con columnas numéricas normalizadas
        - Objeto StandardScaler fitted
    """
    df_normalized = df.copy()
    scaler = StandardScaler()
    df_normalized[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    return df_normalized, scaler


def apply_pca(
    df_normalized: pd.DataFrame, n_components: int | None = None
) -> tuple[pd.DataFrame, PCA, dict, pd.DataFrame]:
    """
    Aplica PCA al DataFrame normalizado.

    Args:
        df_normalized: DataFrame con datos normalizados (solo columnas numéricas)
        n_components: Número de componentes a conservar.
                      Si None, calcula todas las componentes posibles.

    Returns:
        Tuple con:
        - DataFrame con componentes principales
        - Objeto PCA fitted
        - Diccionario con métricas (varianza explicada, etc.)
        - DataFrame con loadings (contribución de variables a componentes)
    """
    # Determinar número máximo de componentes
    max_components = min(df_normalized.shape[0], df_normalized.shape[1])
    if n_components is None:
        n_components = max_components
    elif n_components > max_components:
        n_components = max_components

    pca = PCA(n_components=n_components)
    components = pca.fit_transform(df_normalized)

    # Crear DataFrame con componentes
    component_names = [f"PC{i+1}" for i in range(n_components)]
    df_components = pd.DataFrame(components, columns=component_names)

    # Calcular loadings (contribución de variables a componentes)
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=component_names,
        index=df_normalized.columns,
    )

    # Calcular métricas
    metrics = {
        "n_components": n_components,
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "cumulative_variance_ratio": pca.explained_variance_ratio_.cumsum().tolist(),
        "singular_values": pca.singular_values_.tolist(),
    }

    return df_components, pca, metrics, loadings


def run_normalization_and_pca(n_components: int | None = None) -> dict:
    """
    Ejecuta el pipeline completo de normalización y PCA.

    Args:
        n_components: Número de componentes PCA a conservar.
                      Si None, calcula todas las componentes posibles.

    Returns:
        Diccionario con todos los resultados del pipeline:
        - df_normalized: DataFrame normalizado
        - df_pca: DataFrame con componentes PCA
        - pca_metrics: Métricas del PCA
        - numerical_columns: Columnas numéricas utilizadas
    """
    # Cargar dataset limpio
    dataset_path = config.get_dataset_path("processed") + "/cleaned.csv"
    df = pd.read_csv(dataset_path)

    # Identificar columnas numéricas automáticamente
    numerical_columns = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    # Excluir Person ID si está presente
    if "Person ID" in numerical_columns:
        numerical_columns.remove("Person ID")

    # Normalizar variables numéricas
    df_normalized, scaler = normalize_numerical_features(df, numerical_columns)

    # Extraer solo columnas numéricas para PCA
    df_numerical = df_normalized[numerical_columns]

    # Aplicar PCA
    df_pca, pca_model, pca_metrics, loadings = apply_pca(df_numerical, n_components)

    # Guardar dataset normalizado
    normalized_path = config.get_dataset_path("processed") + "/normalized.csv"
    df_normalized.to_csv(normalized_path, index=False)

    # Guardar componentes PCA
    pca_components_path = config.get_dataset_path("processed") + "/pca_components.csv"
    df_pca.to_csv(pca_components_path, index=False)

    # Guardar loadings
    loadings_path = config.get_dataset_path("processed") + "/pca_loadings.csv"
    loadings.to_csv(loadings_path)

    # Guardar métricas PCA
    pca_reports_path = config.get_output_path("reports", "pca")
    Path(pca_reports_path).mkdir(parents=True, exist_ok=True)
    pca_metrics_path = pca_reports_path + "/pca_metrics.json"
    with open(pca_metrics_path, "w") as f:
        json.dump(pca_metrics, f, indent=2)

    # Serializar scaler y modelo PCA
    scaler_path = config.get_dataset_path("processed") + "/scaler.joblib"
    joblib.dump(scaler, scaler_path)

    pca_model_path = config.get_dataset_path("processed") + "/pca_model.joblib"
    joblib.dump(pca_model, pca_model_path)

    # Retornar resultados
    results = {
        "df_normalized": df_normalized,
        "df_pca": df_pca,
        "pca_metrics": pca_metrics,
        "loadings": loadings,
        "numerical_columns": numerical_columns,
    }

    return results
