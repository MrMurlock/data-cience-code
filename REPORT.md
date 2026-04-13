# Sleep Health and Lifestyle - Reporte de Análisis Exploratorio

---

## 1. Vista General del Dataset

| Métrica | Valor |
|---------|-------|
| Total de Filas | 374 |
| Total de Columnas | 13 |
| Duplicados | 0 |

---

## 2. Calidad de Datos

### Valores Nulos

| Columna | Nulos |
|---------|-------|
| Person ID | 0 |
| Gender | 0 |
| Age | 0 |
| Occupation | 0 |
| Sleep Duration | 0 |
| Quality of Sleep | 0 |
| Physical Activity Level | 0 |
| Stress Level | 0 |
| BMI Category | 0 |
| Blood Pressure | 0 |
| Heart Rate | 0 |
| Daily Steps | 0 |
| **Sleep Disorder** | **219** |

> **Nota:** Los valores nulos en Sleep Disorder representan "None" (ausencia de desorden), no datos faltantes.

### Tipos de Datos

| Columna | Tipo |
|---------|------|
| Person ID | int64 |
| Gender | string |
| Age | int64 |
| Occupation | string |
| Sleep Duration | float64 |
| Quality of Sleep | int64 |
| Physical Activity Level | int64 |
| Stress Level | int64 |
| BMI Category | string |
| Blood Pressure | string |
| Heart Rate | int64 |
| Daily Steps | int64 |
| Sleep Disorder | string |

---

## 3. Estadísticas Numéricas

| Estadístico | Age | Sleep Duration | Quality of Sleep | Physical Activity Level | Stress Level | Heart Rate | Daily Steps |
|-------------|-----|----------------|------------------|------------------------|---------------|------------|-------------|
| **count** | 374.0 | 374.0 | 374.0 | 374.0 | 374.0 | 374.0 | 374.0 |
| **mean** | 42.18 | 7.13 | 7.31 | 59.17 | 5.39 | 70.17 | 6816.84 |
| **std** | 8.67 | 0.80 | 1.20 | 20.83 | 1.77 | 4.14 | 1617.92 |
| **min** | 27 | 5.80 | 4 | 30 | 3 | 65 | 3000 |
| **25%** | 35.25 | 6.40 | 6 | 45 | 4 | 68 | 5600 |
| **50%** | 43.0 | 7.20 | 7 | 60 | 5 | 70 | 7000 |
| **75%** | 50.0 | 7.80 | 8 | 75 | 7 | 72 | 8000 |
| **max** | 59 | 8.50 | 9 | 90 | 8 | 86 | 10000 |

---

## 4. Estadísticas Categóricas

### Gender

| Género | Cantidad |
|--------|----------|
| Male | 189 |
| Female | 185 |

### Occupation

| Ocupación | Cantidad |
|-----------|----------|
| Nurse | 73 |
| Doctor | 71 |
| Engineer | 63 |
| Lawyer | 47 |
| Teacher | 40 |
| Accountant | 37 |
| Salesperson | 32 |
| Software Engineer | 4 |
| Scientist | 4 |
| Sales Representative | 2 |
| Manager | 1 |

### BMI Category

| Categoría BMI | Cantidad |
|---------------|----------|
| Normal | 195 |
| Overweight | 148 |
| Normal Weight | 21 |
| Obese | 10 |

### Blood Pressure

| Presión Arterial | Cantidad |
|------------------|----------|
| 130/85 | 99 |
| 125/80 | 65 |
| 140/95 | 65 |
| 120/80 | 45 |
| 115/75 | 32 |
| 135/90 | 27 |
| 140/90 | 4 |
| 125/82 | 4 |
| 132/87 | 3 |
| 128/85 | 3 |
| Otros (≤2) | 27 |

### Sleep Disorder

| Desorden del Sueño | Cantidad |
|--------------------|----------|
| None | 219 |
| Sleep Apnea | 78 |
| Insomnia | 77 |

---

## 5. Observaciones Clave

1. **Tamaño del Dataset**: 374 individuos, 13 características
2. **Sin Duplicados**: Los datos están limpios de filas duplicadas
3. **Valores Faltantes**: Solo `Sleep Disorder` tiene 219 nulos (58.6%), representando "None" (sin desorden)
4. **Rango de Edad**: 27-59 años (promedio: 42.2)
5. **Duración del Sueño**: 5.8-8.5 horas (promedio: 7.1)
6. **Calidad del Sueño**: Escala 4-9 (promedio: 7.3)
7. **Distribución BMI**: Mayormente Normal (52%) y Overweight (40%)
8. **Distribución de Desordenes**: 58.6% ninguno, 20.9% Apnea, 20.5% Insomnio

---

## 6. Limpieza de Datos

### Transformaciones Aplicadas

| # | Transformación | Descripción |
|---|--------------|-------------|
| 1 | Separar Presión Arterial | `Blood Pressure` → `bp_systolic` + `bp_diastolic` |
| 2 | Completar Sleep Disorder | NaN → "None" (ausencia de desorden) |
| 3 | Estandarizar BMI | "Normal Weight" → "Normal" |
| 4 | Crear Grupo de Edad | 27-35 (Young), 36-45 (Middle), 46-59 (Senior) |
| 5 | Crear Grupo de Calidad | 4-5 (Low), 6-7 (Medium), 8-9 (High) |
| 6 | Convertir a Category | Gender, Occupation, BMI, Sleep Disorder, age_group, quality_group |

### Antes vs Después

| Métrica | Antes | Después |
|---------|-------|---------|
| Filas | 374 | 374 |
| Columnas | 13 | 17 |
| Nulos (Sleep Disorder) | 219 | 0 |

### Nuevas Columnas Agregadas

| Columna | Tipo | Descripción |
|---------|------|-------------|
| bp_systolic | int64 | Presión arterial sistólica |
| bp_diastolic | int64 | Presión arterial diastólica |
| age_group | category | Young/Middle/Senior |
| quality_group | category | Low/Medium/High |

---

## 7. Visualizaciones

Todas las visualizaciones guardadas en: `src/output/figures/`

---

### 7.1 Dataset RAW (Antes de Limpieza)

#### Histogramas (Distribuciones Numéricas)

| Archivo | Descripción |
|---------|-------------|
| `raw_hist_Age.png` | Distribución de edad (27-59 años) |
| `raw_hist_Sleep_Duration.png` | Duración del sueño (5.8-8.5 horas) |
| `raw_hist_Heart_Rate.png` | Distribución de frecuencia cardíaca |
| `raw_hist_Daily_Steps.png` | Distribución de pasos diarios |

#### Bar Chart Ordinal (Quality of Sleep - ordinal)

| Archivo | Descripción |
|---------|-------------|
| `raw_bar_ordinal_Quality_of_Sleep.png` | Calidad del sueño (escala 4-9, ordinal) |

![Age Histogram](/src/output/figures/raw_hist_Age.png)
![Sleep Duration Histogram](/src/output/figures/raw_hist_Sleep_Duration.png)
![Quality of Sleep Bar](/src/output/figures/raw_bar_ordinal_Quality_of_Sleep.png)
![Heart Rate Histogram](/src/output/figures/raw_hist_Heart_Rate.png)
![Daily Steps Histogram](/src/output/figures/raw_hist_Daily_Steps.png)

#### Boxplots (Detección de Outliers)

| Archivo | Descripción |
|---------|-------------|
| `raw_box_Age.png` | Boxplot de edad |
| `raw_box_Sleep_Duration.png` | Boxplot de duración del sueño |
| `raw_box_Quality_of_Sleep.png` | Boxplot de calidad del sueño |
| `raw_box_Heart_Rate.png` | Boxplot de frecuencia cardíaca |
| `raw_box_Daily_Steps.png` | Boxplot de pasos diarios |

![Age Boxplot](/src/output/figures/raw_box_Age.png)
![Sleep Duration Boxplot](/src/output/figures/raw_box_Sleep_Duration.png)
![Quality of Sleep Boxplot](/src/output/figures/raw_box_Quality_of_Sleep.png)
![Heart Rate Boxplot](/src/output/figures/raw_box_Heart_Rate.png)
![Daily Steps Boxplot](/src/output/figures/raw_box_Daily_Steps.png)

#### Pie Charts (Distribuciones Categóricas)

| Archivo | Descripción |
|---------|-------------|
| `raw_pie_Gender.png` | Distribución de género |
| `raw_pie_BMI_Category.png` | Distribución de categoría BMI |
| `raw_pie_Sleep_Disorder.png` | Distribución de desorden del sueño |

![Gender Pie Chart](/src/output/figures/raw_pie_Gender.png)
![BMI Category Pie Chart](/src/output/figures/raw_pie_BMI_Category.png)
![Sleep Disorder Pie Chart](/src/output/figures/raw_pie_Sleep_Disorder.png)

---

### 7.2 Dataset CLEANED (Después de Limpieza)

#### Histogramas (Distribuciones Numéricas)

| Archivo | Descripción |
|---------|-------------|
| `cleaned_hist_Age.png` | Distribución de edad |
| `cleaned_hist_Sleep_Duration.png` | Duración del sueño |
| `cleaned_hist_Heart_Rate.png` | Frecuencia cardíaca |
| `cleaned_hist_Daily_Steps.png` | Pasos diarios |
| `cleaned_hist_bp_systolic.png` | Presión arterial sistólica |
| `cleaned_hist_bp_diastolic.png` | Presión arterial diastólica |

#### Bar Chart Ordinal (Quality of Sleep - ordinal)

| Archivo | Descripción |
|---------|-------------|
| `cleaned_bar_ordinal_Quality_of_Sleep.png` | Calidad del sueño (escala 4-9, ordinal) |

![Systolic Blood Pressure Histogram](/src/output/figures/cleaned_hist_bp_systolic.png)
![Diastolic Blood Pressure Histogram](/src/output/figures/cleaned_hist_bp_diastolic.png)
![Quality of Sleep Bar](/src/output/figures/cleaned_bar_ordinal_Quality_of_Sleep.png)

#### Boxplots

| Archivo | Descripción |
|---------|-------------|
| `cleaned_box_Age.png` | Boxplot de edad |
| `cleaned_box_Sleep_Duration.png` | Boxplot de duración del sueño |
| `cleaned_box_Quality_of_Sleep.png` | Boxplot de calidad del sueño |
| `cleaned_box_Heart_Rate.png` | Boxplot de frecuencia cardíaca |
| `cleaned_box_Daily_Steps.png` | Boxplot de pasos diarios |
| `cleaned_box_bp_systolic.png` | Boxplot de presión sistólica |
| `cleaned_box_bp_diastolic.png` | Boxplot de presión diastólica |

![Systolic Blood Pressure Boxplot](/src/output/figures/cleaned_box_bp_systolic.png)
![Diastolic Blood Pressure Boxplot](/src/output/figures/cleaned_box_bp_diastolic.png)

#### Pie Charts (Distribuciones Categóricas)

| Archivo | Descripción |
|---------|-------------|
| `cleaned_pie_Gender.png` | Distribución de género |
| `cleaned_pie_BMI_Category.png` | Categoría BMI (estandarizada) |
| `cleaned_pie_Sleep_Disorder.png` | Desorden del sueño (sin nulos) |
| `cleaned_pie_age_group.png` | Grupos de edad |
| `cleaned_pie_quality_group.png` | Grupos de calidad |

![Cleaned Pie Sleep Disorder](/src/output/figures/cleaned_pie_Sleep_Disorder.png)
![Age Group Pie Chart](/src/output/figures/cleaned_pie_age_group.png)
![BMI Category Pie Chart](/src/output/figures/cleaned_pie_BMI_Category.png)
![Quality Group Pie Chart](/src/output/figures/cleaned_pie_quality_group.png)

#### Matriz de Correlación

| Archivo | Descripción |
|---------|-------------|
| `cleaned_correlation_heatmap.png` | Correlación entre todas las variables numéricas |

![Correlation Heatmap](/src/output/figures/cleaned_correlation_heatmap.png)

---

### 7.3 Gráficos Bivariados/Trivariados (CLEANED)

Análisis de relaciones entre variables del dataset limpio.

#### Gráficos de Relaciones

| Archivo | Descripción |
|---------|-------------|
| `cleaned_scatter_Sleep_Duration_Quality_of_Sleep.png` | Sleep Duration vs Quality of Sleep |
| `cleaned_box_Sleep_Disorder_Sleep_Duration.png` | Sleep Disorder vs Sleep Duration |
| `cleaned_scatter_Age_Sleep_Duration_by_Sleep_Disorder.png` | Age vs Sleep Duration por Sleep Disorder |
| `cleaned_bar_Sleep_Disorder_by_Gender.png` | Sleep Disorder by Gender |

![Scatter Duration vs Quality](/src/output/figures/cleaned_scatter_Sleep_Duration_Quality_of_Sleep.png)
![Boxplot Disorder vs Duration](/src/output/figures/cleaned_box_Sleep_Disorder_Sleep_Duration.png)
![Scatter Age Disorder](/src/output/figures/cleaned_scatter_Age_Sleep_Duration_by_Sleep_Disorder.png)
![Bar Disorder Gender](/src/output/figures/cleaned_bar_Sleep_Disorder_by_Gender.png)

