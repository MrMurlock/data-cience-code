
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
PYTHONPATH=src poetry run python src/data_cience_code/main.py
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
│   │   │   └── visualization.py   # Gráficos
│   │   ├── models/                  # (futuro)
│   │   │   ├── classification/
│   │   │   └── clustering/
│   │   ├── utils/
│   │   │   ├── helpers.py          # Funciones auxiliares
│   │   │   └── constants.py        # Constantes del proyecto
│   │   └── main.py                  # Pipeline principal
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
│   └── visualization.py # Gráficos y visualizaciones
├── models/
│   ├── classification/   # Modelos de clasificación (futuro)
│   └── clustering/       # Modelos de clustering (futuro)
├── utils/
│   ├── helpers.py        # Funciones auxiliares
│   └── constants.py      # Constantes del proyecto
└── main.py               # Pipeline principal
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

## Fase 3: Modelado (models/*) - FUTURO

### Classification (models/classification/)
- Preparar datos para clasificación
- Entrenar modelos
- Evaluar rendimiento

### Clustering (models/clustering/)
- Preparar datos para clustering
- Determinar número óptimo de clusters
- Analizar segmentos

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