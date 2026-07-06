
# Guía de Trabajo - Proyecto de Ciencia de Datos

## Setup Inicial

### Instalación de Dependencias

```bash
poetry install
```

### Activar Entorno

```bash
poetry env activate
```

### Agregar Nuevos Paquetes

```bash
poetry add package_name
```

### Ejecutar Scripts

```bash
# Pipeline EDA genérico
PYTHONPATH=src poetry run python src/data_cience_code/main.py

# Pipeline completo Sleep Disorder (EDA + Clustering)
PYTHONPATH=src poetry run python -m data_cience_code.pipelines.sleep_disorder_analysis.main

# Solo EDA Sleep Disorder
PYTHONPATH=src poetry run python -m data_cience_code.pipelines.sleep_disorder_analysis.eda_pipeline

# Solo clustering Sleep Disorder
PYTHONPATH=src poetry run python -m data_cience_code.pipelines.sleep_disorder_analysis.clustering_pipeline
```

---

## Estructura del Proyecto

```
data-cience-code/                    # Root del proyecto (contiene Poetry)
├── src/
│   ├── data_cience_code/           # Código fuente Python
│   │   ├── __init__.py
│   │   ├── config/
│   │   │   └── config.py           # Paths y constantes globales
│   │   ├── data/
│   │   │   ├── load.py             # Carga de datos
│   │   │   ├── validate.py         # Validación de calidad
│   │   │   └── preprocess.py       # Limpieza y transformación
│   │   ├── eda/
│   │   │   ├── exploratory.py      # Análisis estadístico
│   │   │   ├── visualization.py   # Gráficos
│   │   │   ├── viz_bivariate.py
│   │   │   ├── viz_cleaned.py
│   │   │   └── viz_raw.py
│   │   ├── models/                  # CÓDIGO GENÉRICO REUTILIZABLE
│   │   │   ├── base/               # Clases base abstractas
│   │   │   │   ├── __init__.py
│   │   │   │   └── clustering_base.py
│   │   │   ├── clustering/
│   │   │   │   ├── kmeans.py
│   │   │   │   ├── hierarchical.py
│   │   │   │   ├── dbscan.py
│   │   │   │   ├── comparison.py
│   │   │   │   ├── cluster_analysis.py
│   │   │   │   └── utils.py
│   │   │   ├── dimensionality_reduction/
│   │   │   │   ├── pca.py
│   │   │   │   └── visualization_pca.py
│   │   │   └── classification/     # Futuro: reglas de asociación, árboles
│   │   ├── pipelines/             # PIPELINES ESPECÍFICOS POR CASO DE ESTUDIO
│   │   │   ├── sleep_disorder_analysis/  # Caso actual: Sleep Disorder
│   │   │   │   ├── main.py         # Pipeline principal EDA + clustering
│   │   │   │   ├── eda_pipeline.py  # Pipeline EDA específico
│   │   │   │   ├── clustering_pipeline.py  # Pipeline clustering específico
│   │   │   │   └── visualization.py  # Visualizaciones específicas
│   │   │   └── template/          # Plantilla para futuros casos
│   │   │       ├── __init__.py
│   │   │       └── pipeline_template.py
│   │   ├── utils/
│   │   │   ├── helpers.py          # Funciones auxiliares
│   │   │   └── constants.py        # Constantes del proyecto
│   │   └── main.py                  # Pipeline EDA genérico
│   ├── dataset/
│   │   ├── raw/
│   │   └── processed/
│   └── output/
│       ├── figures/
│       └── reports/
├── tests/                          # Tests del proyecto
├── pyproject.toml                  # Configuración Poetry
└── AGENTS.md                       # Esta guía
```

### Imports Python dentro del Package

Para código Python dentro de `src/data_cience_code/`, usa imports relativos:

```python
# Desde src/data_cience_code/main.py
from config import config
from data import load, validate
from eda import exploratory, visualization

# Uso
dataset_path = config.get_dataset_path("raw") + "/original.csv"
```

**Nota importante:** El proyecto usa `package-mode = false` en `pyproject.toml`, por lo que NO es un package instalable. Los imports deben ser relativos al archivo.

**Reglas:**

- Ejecuta scripts desde la raíz del proyecto (`data-cience-code/`)
- Los imports son relativos al package `data_cience_code/`
- Nunca uses `cd` para cambiar de directorio antes de ejecutar

---

> **Nota:** Antes de iniciar el análisis, revisa `DATASET.md` para entender la estructura y descripción del dataset.
> **Nota:** El reporte de análisis (`REPORT.md`) está en español.

## Configuración de Paths para Archivos No-Python

### Variables Globales en config/config.py

Para archivos de datos (CSV, imágenes, etc.), usa paths absolutos definidos en `config/config.py`:

```python
# src/data_cience_code/config/config.py
# El usuario define sus paths absolutos aquí:

DATASET_FOLDER_ABS_PATH = "/home/murlock/Descargas/data-cience-code/src/dataset"
OUTPUT_FOLDER_ABS_PATH = "/home/murlock/Descargas/data-cience-code/src/output"

# Dataset paths
# absolute path for raw and processed folders
# dataset_type: "raw" or "processed"
def get_dataset_path(dataset_type: str) -> str:
    switch = {
        "raw": f"{DATASET_FOLDER_ABS_PATH}/raw",
        "processed": f"{DATASET_FOLDER_ABS_PATH}/processed"
    }
    return switch[dataset_type]

# Output paths
# absolute path for figures and reports folders
# output_type: "figures" or "reports"
def get_output_path(output_type: str) -> str:
    switch = {
        "figures": f"{OUTPUT_FOLDER_ABS_PATH}/figures",
        "reports": f"{OUTPUT_FOLDER_ABS_PATH}/reports"
    }
    return switch[output_type]
```

### Uso en cualquier script

```python
# En main.py, notebooks
import config.config as config

# Leer dataset
dataset_path = config.get_dataset_path("raw") + "/original.csv"
df = pd.read_csv(dataset_path)

# Guardar figura
output_path = config.get_output_path("figures") + "/figure.png"
plt.savefig(output_path)
```

### Diferencia clave

| Tipo | Método | Ejemplo |
|------|--------|---------|
| **Código Python** | Import relativo | `import config.config as config` |
| **Archivos (CSV, PNG, etc)** | Path absoluto vía config | `pd.read_csv(config.get_dataset_path("raw") + "/data.csv")` |

---

## Estructura de Módulos

```
src/data_cience_code/
├── config/
│   └── config.py           # Paths y constantes globales
├── data/
│   ├── load.py            # Carga de datos (CSV, Excel, etc.)
│   ├── validate.py        # Validación de calidad de datos
│   └── preprocess.py      # Limpieza y transformación básica
├── eda/
│   ├── exploratory.py    # Análisis estadístico descriptivo
│   ├── visualization.py # Gráficos y visualizaciones
│   ├── viz_bivariate.py
│   ├── viz_cleaned.py
│   └── viz_raw.py
├── models/                 # CÓDIGO GENÉRICO REUTILIZABLE
│   ├── base/              # Clases base para visualizaciones genéricas
│   ├── clustering/        # Algoritmos de clustering genéricos
│   ├── dimensionality_reduction/  # PCA y reducción de dimensionalidad
│   └── classification/    # Futuro: clasificación, reglas de asociación
├── pipelines/             # PIPELINES ESPECÍFICOS POR CASO DE ESTUDIO
│   ├── sleep_disorder_analysis/  # Pipeline actual (Sleep Disorder)
│   └── template/          # Plantilla para nuevos pipelines
├── utils/
│   ├── helpers.py        # Funciones auxiliares
│   └── constants.py      # Constantes del proyecto
└── main.py               # Pipeline EDA genérico
```

### Reglas de Módulos

- **Todos los scripts son importables**, no ejecutables directamente
- **Imports relativos** dentro del paquete `data_cience_code`
- Cada módulo tiene una **función principal** `run()` que retorna el resultado
- **Type hints** en todas las funciones públicas

---

## Fase 1: Análisis Exploratorio de Datos (EDA)

### Objetivos
- Comprender la estructura y características del dataset
- Identificar patrones, anomalías y relaciones entre variables
- Generar hipótesis iniciales

### Checklist EDA

#### 1. Carga y Validación (data/load.py + data/validate.py)
- [ ] Cargar dataset desde `dataset/raw/` usando `config.get_dataset_path("raw")`
- [ ] Verificar dimensiones (filas × columnas)
- [ ] Inspeccionar primeras/últimas filas (`head()`, `tail()`)
- [ ] Validar tipos de datos (`dtypes`)

#### 2. Análisis de Calidad de Datos (data/validate.py)
- [ ] Contar valores nulos por columna
- [ ] Calcular porcentaje de missing values
- [ ] Detectar duplicados (`duplicated()`)
- [ ] Identificar valores constantes o de baja varianza

#### 3. Análisis Estadístico Descriptivo (eda/exploratory.py)
- [ ] Resumen estadístico numérico (`describe()`)
- [ ] Distribución de variables categóricas (`value_counts()`)
- [ ] Medidas de tendencia central y dispersión
- [ ] Identificar outliers (IQR, Z-score)

#### 4. Visualizaciones Obligatorias (eda/visualization.py)
- [ ] Histogramas de variables numéricas
- [ ] Boxplots para detectar outliers
- [ ] Correlación entre variables numéricas (heatmap)
- [ ] Gráficos de barras para categóricas
- [ ] Pairplot para relaciones bivariadas (si aplica)

---

## Fase 2: Limpieza de Datos (data/preprocess.py)

### Estrategias por Tipo de Problema

#### Valores Faltantes
| Estrategia | Cuándo Usar | Implementación |
|------------|-------------|----------------|
| Eliminación | <5% missing, aleatorios | `dropna()` |
| Imputación media/mediana | Numéricos, distribución normal | `fillna(df.mean())` |
| Imputación por moda | Categóricos | `fillna(df.mode()[0])` |
| Forward/Backward fill | Series temporales | `fillna(method='ffill')` |

#### Duplicados
```python
# Detectar
print(f"Duplicados: {df.duplicated().sum()}")

# Eliminar
df_clean = df.drop_duplicates()
```

#### Outliers
```python
# Método IQR
Q1 = df['columna'].quantile(0.25)
Q3 = df['columna'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['columna'] < Q1 - 1.5*IQR) | (df['columna'] > Q3 + 1.5*IQR)]
```

#### Corrección de Tipos de Datos
```python
# Convertir a datetime
df['fecha'] = pd.to_datetime(df['fecha'])

# Convertir a categórico
df['categoria'] = df['categoria'].astype('category')

# Convertir a numérico
df['numero'] = pd.to_numeric(df['numero'], errors='coerce')
```

---

## Fase 3: Modelado (models/*)

### Arquitectura de Código

**models/** contiene código genérico reutilizable:
- Funciones de entrenamiento genéricas (train_kmeans, train_hierarchical, train_dbscan)
- Visualizaciones genéricas (plot_clusters_3d, plot_biplot_3d)
- Clases base para extender (ClusterVisualizer en models/base/)
- Sin referencias a datasets específicos

**pipelines/** contiene código específico de cada caso de estudio:
- pipelines/sleep_disorder_analysis/ para el dataset actual
- pipelines/template/ como plantilla para nuevos análisis
- Cada pipeline usa funciones genéricas de models/
- Visualizaciones específicas del dataset (ej: overlay con Sleep Disorder)

### Classification (models/classification/)
- Preparar datos para clasificación
- Entrenar modelos
- Evaluar rendimiento

### Clustering (models/clustering/)
- Preparar datos para clustering
- Determinar número óptimo de clusters
- Analizar segmentos

### Crear Nuevo Pipeline

Para crear un nuevo pipeline para un dataset diferente:

1. Copiar `pipelines/template/` a `pipelines/your_analysis/`
2. Implementar funciones específicas del dataset
3. Crear visualizaciones específicas si es necesario
4. Usar funciones genéricas de `models/` para el análisis
5. Crear `main.py` que orqueste el pipeline completo

---

## Convenciones de Código

### Organización de Imports
```python
# 1. Librerías estándar
import os
from pathlib import Path

# 2. Librerías de terceros
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 3. Módulos del proyecto
from data_cience_code.config import config
from data_cience_code.data import load, validate, preprocess
from data_cience_code.eda import exploratory, visualization
from data_cience_code.models import clustering  # Para código genérico
from data_cience_code.pipelines.sleep_disorder_analysis import main  # Para pipeline específico
```

### Configuración de Visualizaciones
```python
# Estilo consistente
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Tamaño por defecto
plt.rcParams['figure.figsize'] = (12, 6)
```

### Guardar Resultados
```python
# Estructura de carpetas sugerida
DATASET_RAW = "dataset/raw"
DATASET_PROCESSED = "dataset/processed"
OUTPUT_FIGURES = "output/figures"
OUTPUT_REPORTS = "output/reports"
```

---

## Comandos Frecuentes

| Comando | Descripción |
|---------|-------------|
| `poetry install` | Instalar dependencias |
| `poetry add <pkg>` | Agregar paquete |
| `poetry run python <script>` | Ejecutar script |
| `poetry env info` | Info del entorno |

---

## Recursos Recomendados

- **pandas**: Manipulación y análisis de datos
- **numpy**: Operaciones numéricas
- **matplotlib/seaborn**: Visualizaciones
- **jupyter**: Notebooks interactivos