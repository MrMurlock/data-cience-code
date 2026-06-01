import sys
from pathlib import Path

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.clustering import generate_comparison_report

if __name__ == "__main__":
    print("Generando tabla comparativa de métricas de clustering...")

    # Generar reporte comparativo
    df_comparison = generate_comparison_report(
        k_values=[4, 6, 10, 14],
        dbscan_eps=0.7117,
        dbscan_min_samples=5,
    )

    print("\n=== Tabla Comparativa de Métricas ===")
    print(df_comparison.to_string(index=False))

    print("\n✅ Tabla comparativa generada exitosamente")
