# Ayudantía Ciencia de Datos

Trabajo de ejemplo para el análisis de un dataset sintético sobre salud del sueño y estilo de vida.

El proyecto utiliza:

* Python para procesamiento, análisis y generación de artefactos.
* Markdown para los reportes.
* Marp para las presentaciones.
* `uv` para la gestión del entorno y dependencias.

El código Python está organizado en pipelines según el tipo de trabajo que se desea realizar. Los pipelines pueden realizar procesamiento de datos, análisis estadístico y generación de artefactos como gráficos y tablas.

Los gráficos y tablas generados por los pipelines son consumidos posteriormente por los reportes y presentaciones. Python genera los artefactos (tablas, métricas e imágenes) y los datos necesarios, pero el contenido narrativo de los documentos se escribe en Markdown como template y se renderiza a su versión final a través del mismo pipeline.

## Estructura del proyecto

En la raíz del proyecto se encuentra `dataset`, que contiene los datos utilizados por los distintos trabajos:

* `dataset/raw`: datos originales, sin modificaciones.
* `dataset/processed`: datos transformados o preparados por los pipelines.

Los artefactos generados por los pipelines se guardan en `artifacts` (por ejemplo, `artifacts/eda` con `tables`, `metrics` e `images`). Los documentos fuente editables viven en `reports/eda` (`EDA.md`, `presentation.md`); sus versiones renderizadas (`*.final.md`), el PPTX exportado con Marp y los reportes de profiling en HTML se guardan como outputs en `output/eda`.

`DATASET.md` contiene la documentación del dataset, incluyendo su origen, variables y descripción de las columnas.

En `src/data_cience_code` se encuentra el código Python:

* `config`: constantes y utilidades de configuración utilizadas por defecto por los pipelines.
* `utils`: funciones, diccionarios y clases de utilidad general reutilizables entre pipelines (artefactos, render de documentos, gráficos, profiling).
* `pipelines`: contiene un pipeline por tipo de trabajo. Cada pipeline tiene un `__main__.py` como entrypoint, que importa desde `pipeline.py` la función principal de ejecución. La implementación y los módulos específicos del pipeline viven dentro de su propio directorio.
* `docs`: documentación referencial y especificaciones utilizadas por el proyecto, tanto por usuarios como por agentes de desarrollo.

La estructura actual es:

```text
.
├── artifacts
│   └── eda
│       ├── images
│       ├── metrics
│       └── tables
├── dataset
│   ├── processed
│   └── raw
├── docs
│   ├── 0.DATASET.md
│   └── 1.EDA.md
├── pyproject.toml
├── README.md
├── output
│   └── eda
│       ├── EDA.final.md / presentation.final.md
│       ├── presentation.final.pptx
│       └── profiling (HTML)
├── reports
│   └── eda
│       ├── EDA.md
│       └── presentation.md
├── src
│   └── data_cience_code
│       ├── config
│       │   └── const.py
│       ├── __init__.py
│       ├── pipelines
│       │   └── eda
│       │       ├── __main__.py
│       │       ├── bivariate.py
│       │       ├── config_eda.py
│       │       ├── pipeline.py
│       │       ├── preprocessing.py
│       │       ├── quality.py
│       │       ├── render_eda.py
│       │       └── univariate.py
│       └── utils
│           ├── artifacts.py
│           ├── io.py
│           ├── plots.py
│           ├── profilling.py
│           ├── render.py
│           └── tables.py
├── tests
│   └── __init__.py
└── uv.lock
```

## Documentación

La documentación específica de cada trabajo se encuentra en `docs`.

Para el análisis exploratorio de datos:

* `docs/1.EDA.md`: especificación del análisis EDA, sus artefactos y los productos documentales asociados.
* `DATASET.md`: documentación del dataset utilizado como fuente.

Los documentos de `docs` funcionan como referencia para la implementación de los pipelines y para la elaboración de los reportes y presentaciones.

## Comandos útiles

### Inicializar el proyecto

```bash
uv sync
```

### Instalar un paquete

```bash
uv add pandas
```

### Desinstalar un paquete

```bash
uv remove pandas
```

### Ejecutar el pipeline EDA

El pipeline hace todo el flujo: preprocessing, análisis (univariado y bivariado), generación de artefactos y render de los documentos finales:

```bash
uv run python -m data_cience_code.pipelines.eda
```

Variantes útiles:

```bash
# solo render de los documentos, sin recalcular el análisis
uv run python -m data_cience_code.pipelines.eda --render-only

# solo análisis, sin render de los documentos
uv run python -m data_cience_code.pipelines.eda --no-render

# además exporta la presentación a PPTX (requiere Marp CLI instalado)
uv run python -m data_cience_code.pipelines.eda --render-only --pptx
```

La conversión de la presentación requiere Marp CLI (opcional); el pipeline siempre genera el Markdown final de Marp (`output/eda/presentation.final.md`).

Los documentos editables (`reports/eda/`) y los outputs generados (`output/eda/`) se mantienen separados: los `*.final.md`, el `presentation.final.pptx` y los HTML de profiling de `output/eda/` son productos del pipeline y no deben editarse manualmente.
