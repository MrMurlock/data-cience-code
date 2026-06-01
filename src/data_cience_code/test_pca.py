import sys
from pathlib import Path

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.dimensionality_reduction import run_normalization_and_pca

if __name__ == "__main__":
    print("Ejecutando pipeline de normalización y PCA...")
    results = run_normalization_and_pca()

    print("\n=== Resultados del Pipeline ===")
    print(f"Columnas numéricas utilizadas: {results['numerical_columns']}")
    print(f"Shape dataset normalizado: {results['df_normalized'].shape}")
    print(f"Shape componentes PCA: {results['df_pca'].shape}")
    print(f"Shape loadings: {results['loadings'].shape}")
    print(f"Número de componentes PCA: {results['pca_metrics']['n_components']}")

    print("\n=== Varianza Explicada por Componente ===")
    for i, (var_ratio, cum_var) in enumerate(
        zip(
            results["pca_metrics"]["explained_variance_ratio"],
            results["pca_metrics"]["cumulative_variance_ratio"],
        )
    ):
        print(f"PC{i+1}: {var_ratio:.4f} ({cum_var:.4f} acumulado)")

    print("\n=== Loadings (Contribución de Variables a Componentes) ===")
    print(results["loadings"].round(4))

    print("\n=== Archivos Generados ===")
    print("- src/dataset/processed/normalized.csv")
    print("- src/dataset/processed/pca_components.csv")
    print("- src/dataset/processed/pca_loadings.csv")
    print("- src/dataset/processed/scaler.joblib")
    print("- src/dataset/processed/pca_model.joblib")
    print("- src/output/reports/pca/pca_metrics.json")

    print("\n✅ Pipeline completado exitosamente")
