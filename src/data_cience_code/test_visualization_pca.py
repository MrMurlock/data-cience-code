import sys
from pathlib import Path

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.dimensionality_reduction.visualization_pca import run_pca_visualizations

if __name__ == "__main__":
    print("Ejecutando visualizaciones PCA...")
    results = run_pca_visualizations()

    print("\n=== Visualizaciones Generadas ===")
    print(f"Biplot 2D: {results['biplot_2d_path']}")
    print(f"Biplot 3D: {results['biplot_3d_path']}")

    print("\n✅ Visualizaciones completadas exitosamente")
