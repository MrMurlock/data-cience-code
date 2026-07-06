# Análisis de Reglas de Asociación — Sleep Health and Lifestyle

---

## 1. Introducción

### 1.1 ¿Qué son las Reglas de Asociación?

Las reglas de asociación son un método de aprendizaje no supervisado que descubre relaciones entre variables en grandes conjuntos de datos. Se utilizan ampliamente en análisis de mercado (market basket analysis), pero su alcance se extiende a cualquier dominio donde se busquen patrones de co-ocurrencia entre ítems.

Una regla de asociación tiene la forma:

```
{Antecedente} → {Consecuente}
```

que se lee como: "si ocurre el antecedente, entonces es probable que ocurra el consecuente".

### 1.2 Terminología

| Término | Definición |
|---------|------------|
| **Itemset** | Conjunto de uno o más ítems |
| **Itemset frecuente** | Itemset cuya frecuencia supera un umbral mínimo de soporte |
| **Regla** | Implicación de la forma A → C, donde A y C son itemsets disjuntos |
| **Antecedente (LHS)** | Condición o premisa de la regla (left-hand side) |
| **Consecuente (RHS)** | Resultado o conclusión de la regla (right-hand side) |
| **Transacción** | Cada registro individual del conjunto de datos |

### 1.3 Objetivo del Análisis

Aplicar minería de reglas de asociación sobre el dataset **Sleep Health and Lifestyle** para:

- Identificar patrones entre variables de salud, sueño y estilo de vida
- Descubrir factores asociados a trastornos del sueño (Insomnio, Apnea)
- Comparar el rendimiento de los algoritmos **Apriori** y **FP-Growth**
- Generar reglas filtradas por trastorno de sueño y por ocupación

---

## 2. Algoritmos

### 2.1 Apriori

**Principio**: Un itemset es frecuente solo si todos sus subconjuntos también lo son (propiedad *anti-monótona*). El algoritmo genera candidatos de tamaño k a partir de itemsets frecuentes de tamaño k-1 y luego poda aquellos cuyos subconjuntos no son frecuentes.

**Ventajas**:
- Simple y fácil de implementar
- Garantiza encontrar todos los itemsets frecuentes

**Desventajas**:
- Requiere múltiples escaneos de la base de datos
- Generación costosa de candidatos para itemsets largos
- Rendimiento degrada con umbrales de soporte bajos

### 2.2 FP-Growth (Frequent Pattern Growth)

**Principio**: Compress la base de datos en una estructura de árbol compacta llamada **FP-tree** (Frequent Pattern Tree), que preserva la información de itemsets. Luego extrae itemsets frecuentes recursivamente sin generar candidatos explícitos.

**Ventajas**:
- Solo requiere dos escaneos de la base de datos
- No genera candidatos, lo que reduce el costo computacional
- Generalmente más rápido que Apriori en datasets densos

**Desventajas**:
- Mayor uso de memoria para el FP-tree
- Implementación más compleja

### 2.3 Comparación en este Dataset

| Característica | Apriori | FP-Growth |
|----------------|---------|-----------|
| Tiempo total (4 soportes) | 0.0523s | 0.0780s |
| Reglas totales (s ≥ 0.1) | 2932 | 2932 |
| Solapamiento único | 100% | 100% |

Ambos algoritmos producen resultados idénticos en mlxtend (implementación Python pura). Apriori resultó ligeramente más rápido en este dataset, probablemente debido al tamaño reducido (374 transacciones, 41 ítems).

### 2.4 Configuración de Parámetros

- **Umbrales de soporte evaluados**: 0.10, 0.15, 0.20, 0.25
- **Longitud máxima de itemsets**: 3
- **Métrica de poda**: confidence ≥ 0.5

---

## 3. Métricas de Evaluación

Para cada regla se calculan las siguientes métricas:

### 3.1 Soporte (Support)

Frecuencia relativa de la regla en el dataset:

```
support(A → C) = P(A ∩ C) = (transacciones con A y C) / (total transacciones)
```

Mide qué tan frecuente es la regla en la población. Un soporte alto indica que la regla aplica a una porción considerable de los datos.

### 3.2 Confianza (Confidence)

Probabilidad condicional del consecuente dado el antecedente:

```
confidence(A → C) = P(C | A) = support(A ∪ C) / support(A)
```

Mide la fiabilidad de la regla. Una confianza de 0.8 significa que el 80% de las transacciones con A también contienen C.

### 3.3 Lift

Ratio entre la probabilidad observada y la esperada bajo independencia:

```
lift(A → C) = P(C | A) / P(C) = confidence(A → C) / support(C)
```

Interpretación:
- **lift > 1**: asociación positiva (A y C ocurren juntos más de lo esperado)
- **lift = 1**: independencia
- **lift < 1**: asociación negativa

### 3.4 Leverage

Diferencia entre la frecuencia observada y la esperada bajo independencia:

```
leverage(A → C) = support(A ∪ C) - support(A) × support(C)
```

Rango: [-0.25, 0.25]. Valores positivos indican dependencia positiva.

### 3.5 Convicción (Conviction)

Ratio de dependencia direccional:

```
conviction(A → C) = (1 - support(C)) / (1 - confidence(A → C))
```

Rango: [0, ∞). Una convicción alta significa que el consecuente depende fuertemente del antecedente. Valor neutro = 1.

### 3.7 Resumen de Métricas

| Métrica | Rango | Valor Neutro | Interpretación |
|---------|-------|-------------|----------------|
| Soporte | [0, 1] | — | Qué tan frecuente es la combinación A y C en todo el dataset |
| Confianza | [0, 1] | P(C) | De todos los casos con A, en qué proporción también aparece C |
| Lift | [0, ∞) | 1 | Cuántas veces más (o menos) aparecen juntos A y C comparado con azar |
| Leverage | [-1, 1] aprox | 0 | Cuánto más aparecen juntos A y C respecto a lo esperado por azar (en “puntos de probabilidad”) |
| Convicción | [0, ∞) | 1 | Qué tan rara es la situación donde A ocurre pero C no ocurre |

---

## 4. Preparación de Datos

### 4.1 Carga

Se parte del archivo `cleaned.csv` (374 filas x 13 columnas), resultado del pipeline EDA que incluye limpieza y transformación inicial (separación de presión arterial en sistólica/diastólica, creación de grupos etarios y de calidad de sueño).

### 4.2 Discretización de Variables Numéricas

Los puntos de corte se basan en los percentiles REPORT.md del dataset:

| Variable | Bins (etiquetas) | Criterio |
|----------|-------------------|----------|
| **Sleep Duration** | [5.8, 6.4, 7.8, 8.5] → Corto / Medio / Largo | Percentiles 25 y 75 |
| **Stress Level** | [2, 4, 6, 9] → Bajo / Medio / Alto | Percentiles 25 y 75 |
| **Physical Activity** | [29, 45, 75, 91] → Baja / Media / Alta | Percentiles 25 y 75 |
| **Heart Rate** | [64, 67, 72, 87] → Baja / Normal / Alta | Percentiles 25 y 75 |
| **Daily Steps** | [2999, 5600, 8000, 10001] → Sedentario / Moderado / Activo | Percentiles 25 y 75 |
| **BP Systolic** | [0, 119, 129, 200] → Normal / Elevada / Alta | Umbrales clínicos |
| **BP Diastolic** | [0, 79, 89, 200] → Normal / Elevada / Alta | Umbrales clínicos |

### 4.3 Agrupación de Ocupaciones

Las 11 ocupaciones originales se agrupan en 6 categorías funcionales:

| Categoría | Ocupaciones |
|-----------|-------------|
| **Healthcare** | Doctor, Nurse |
| **Technical** | Engineer, Software Engineer, Scientist |
| **Education** | Teacher |
| **Business** | Accountant, Manager |
| **Sales** | Salesperson, Sales Representative |
| **Legal** | Lawyer |

### 4.4 Codificación One-Hot

Las 13 columnas categóricas resultantes se transforman a 41 columnas binarias (formato one-hot), donde cada columna representa un ítem individual (ej. `Gender_Male`, `age_group_Senior`, `BMI Category_Sobrepeso`).

---

## 5. Resultados Globales

### 5.1 Estadísticas Generales

- **Total de transacciones**: 374
- **Total de ítems únicos**: 41
- **Reglas generadas (s ≥ 0.1, max_len=3)**: 2932
- **Solapamiento Apriori / FP-Growth**: 100%

### 5.2 Rendimiento por Algoritmo

| min_support | Reglas Apriori | Reglas FP-Growth | Tiempo Apriori (s) | Tiempo FP-Growth (s) |
|-------------|----------------|------------------|-------------------|---------------------|
| 0.10 | 2932 | 2932 | 0.054 | 0.076 |
| 0.15 | 541 | 541 | 0.033 | 0.046 |
| 0.20 | 100 | 100 | 0.026 | 0.030 |
| 0.25 | 14 | 14 | 0.022 | 0.024 |

### 5.3 Distribución de Métricas

![Support vs Confidence](/src/output/figures/association_rules/apriori_support_vs_confidence.png)

*Figura 1: Relación entre soporte y confianza para todas las reglas. Cada punto representa una regla; el color puede indicar lift.*

![Distribution of Support](/src/output/figures/association_rules/distribution_support.png)

*Figura 2: Distribución de soporte entre las reglas. La mayoría de las reglas tienen soporte entre 0.10 y 0.20.*

![Distribution of Confidence](/src/output/figures/association_rules/distribution_confidence.png)

*Figura 3: Distribución de confianza. La mayoría de las reglas presentan confianza entre 0.50 y 0.80.*

![Distribution of Lift](/src/output/figures/association_rules/distribution_lift.png)

*Figura 4: Distribución de lift. La mayoría de las reglas tienen lift entre 1.0 y 4.0, con algunos valores extremos hasta 9.12.*

### 5.4 Top Reglas por Lift

Las reglas con mayor lift son correlaciones entre presión arterial sistólica y diastólica normales (lift ~9.12), lo que refleja la correlación natural esperada entre ambas mediciones.

![Top Rules Apriori](/src/output/figures/association_rules/top_rules_apriori.png)

*Figura 5: Top 10 reglas de Apriori ordenadas por lift.*

![Top Rules FP-Growth](/src/output/figures/association_rules/top_rules_fp_growth.png)

*Figura 6: Top 10 reglas de FP-Growth ordenadas por lift.*

### 5.5 Comparación de Algoritmos

![Comparison Bar](/src/output/figures/association_rules/comparison_bar.png)

*Figura 7: Comparación del número de reglas generadas por Apriori y FP-Growth en cada umbral de soporte.*

### 5.6 Matriz de Calidad de Reglas

![Rule Quality Matrix](/src/output/figures/association_rules/rule_quality_matrix.png)

*Figura 8: Mapa de calor de métricas (support, confidence, lift, leverage, conviction, zhangs_metric) para reglas seleccionadas.*

### 5.7 Red de Asociaciones

![Network Graph](/src/output/figures/association_rules/network_graph.png)

*Figura 9: Grafo de red de las asociaciones más fuertes. Los nodos representan ítems y las aristas representan reglas con alta confianza y lift. El grosor de la arista indica la fuerza de la asociación.*

### 5.8 Items más Frecuentes en Antecedentes

| Item | Frecuencia |
|------|-----------|
| Presion Arterial Diastólica: Elevada | 250 |
| Calidad: Alta | 224 |
| Presion Arterial Sistólica: Alta | 219 |
| BMI: Normal | 213 |
| SD: Ninguno | 212 |
| Male | 211 |
| Frecuencia cardíaca: Normal | 205 |
| Estrés: Alto | 197 |
| Edad: Medio | 196 |
| Cantidad de pasos: Activo | 186 |

---

## 6. Resultados Filtrados

### 6.1 Reglas con Insomnio (Sleep Disorder = Insomnia)

**12 reglas** encontradas donde el consecuente es Insomnio.

**Patrón dominante**: Sobrepeso + Actividad Física Media o Edad Media.

| # | Regla | Soporte | Confianza | Lift |
|---|-------|---------|-----------|------|
| 1 | Actividad: Media, BMI: Sobrepeso → SD: Insomnio | 15.8% | 83.1% | 4.04 |
| 2 | Edad: Medio, BMI: Sobrepeso → SD: Insomnio | 12.3% | 79.3% | 3.85 |
| 3 | Duracion de sueño: Medio, BMI: Sobrepeso → SD: Insomnio | 13.6% | 78.5% | 3.81 |
| 4 | Presion Arterial Diastólica: Elevada, BMI: Sobrepeso → SD: Insomnio | 10.2% | 67.9% | 3.30 |
| 5 | Actividad: Media, Frecuencia cardíaca: Alta → SD: Insomnio | 10.2% | 66.7% | 3.24 |

![Insomnio - Top Rules](/src/output/figures/association_rules/filter_by/insomnio/top_rules.png)

![Insomnio - Quality Matrix](/src/output/figures/association_rules/filter_by/insomnio/quality_matrix.png)

![Insomnio - Network Graph](/src/output/figures/association_rules/filter_by/insomnio/network_graph.png)

![Insomnio - Support vs Confidence](/src/output/figures/association_rules/filter_by/insomnio/support_vs_confidence.png)

![Insomnio - Distribution Lift](/src/output/figures/association_rules/filter_by/insomnio/distribution_lift.png)

### 6.2 Reglas con Apnea del Sueño (Sleep Disorder = Sleep Apnea)

**27 reglas** encontradas donde el consecuente es Apnea del Sueño.

**Patrón dominante**: Sector Salud (Healthcare) + Presión Arterial Alta o Edad Senior o Actividad Física Alta. La apnea se asocia fuertemente con perfiles de mayor edad y sobrepeso.

| # | Regla | Soporte | Confianza | Lift |
|---|-------|---------|-----------|------|
| 1 | Presion Arterial Diastólica: Alta, Actividad: Alta → SD: Apnea | 16.3% | 91.0% | 4.37 |
| 2 | BMI: Sobrepeso, Actividad: Alta → SD: Apnea | 15.8% | 89.4% | 4.29 |
| 3 | Ocup: Salud, BMI: Sobrepeso → SD: Apnea | 15.8% | 89.4% | 4.29 |
| 4 | Edad: Senior, Actividad: Alta → SD: Apnea | 16.3% | 88.4% | 4.24 |
| 5 | Ocup: Salud, Presion Arterial Diastólica: Alta → SD: Apnea | 16.3% | 88.4% | 4.24 |

![Apnea - Top Rules](/src/output/figures/association_rules/filter_by/apnea/top_rules.png)

![Apnea - Quality Matrix](/src/output/figures/association_rules/filter_by/apnea/quality_matrix.png)

![Apnea - Network Graph](/src/output/figures/association_rules/filter_by/apnea/network_graph.png)

![Apnea - Support vs Confidence](/src/output/figures/association_rules/filter_by/apnea/support_vs_confidence.png)

![Apnea - Distribution Lift](/src/output/figures/association_rules/filter_by/apnea/distribution_lift.png)

### 6.3 Reglas sin Trastorno del Sueño (Sleep Disorder = None)

**244 reglas** encontradas donde el consecuente es la ausencia de trastorno del sueño.

**Patrón dominante**: Ocupación Legal aparece con alta confianza. Edad Joven se asocia fuertemente a ausencia de trastornos, incluso cuando se combina con estrés alto.

| # | Regla | Soporte | Confianza | Lift |
|---|-------|---------|-----------|------|
| 1 | Ocup: Legal → SD: Ninguno, Presion Arterial Sistólica: Alta | 10.4% | 83.0% | 5.09 |
| 2 | Edad: Joven → Ocup: Salud, SD: Ninguno | 16.8% | 76.8% | 3.94 |
| 3 | Actividad: Baja → SD: Ninguno, Cantidad de pasos: Sedentario | 15.2% | 69.5% | 3.88 |
| 4 | Edad: Joven → SD: Ninguno, Estrés: Alto | 17.6% | 80.5% | 3.81 |
| 5 | Cantidad de pasos: Sedentario → SD: Ninguno, Actividad: Baja | 15.2% | 61.3% | 3.58 |

![Sin Trastorno - Top Rules](/src/output/figures/association_rules/filter_by/sin_trastorno/top_rules.png)

![Sin Trastorno - Quality Matrix](/src/output/figures/association_rules/filter_by/sin_trastorno/quality_matrix.png)

![Sin Trastorno - Network Graph](/src/output/figures/association_rules/filter_by/sin_trastorno/network_graph.png)

![Sin Trastorno - Support vs Confidence](/src/output/figures/association_rules/filter_by/sin_trastorno/support_vs_confidence.png)

![Sin Trastorno - Distribution Lift](/src/output/figures/association_rules/filter_by/sin_trastorno/distribution_lift.png)

### 6.4 Reglas con Ocupación en Antecedente

**411 reglas** donde el antecedente incluye el grupo ocupacional.

**Distribución por ocupación**:

| Ocupación | Frecuencia en reglas |
|-----------|---------------------|
| Ocup: Salud | 174 |
| Ocup: Legal | 155 |
| Ocup: Técnica | 80 |
| Ocup: Educación | 1 |
| Ocup: Negocios | 1 |

**Patrones por ocupación**:

- **Legal**: Reglas con muy alta confianza (83-93%). Se asocian a pasos activos, estrés medio, presión sistólica alta, y ausencia de trastorno del sueño.
- **Salud (Healthcare)**: Reglas relacionadas con apnea del sueño, presión arterial y edad senior.
- **Técnica (Technical)**: Reglas con frecuencia cardíaca baja y presión elevada, perfil sedentario.

| # | Regla | Soporte | Confianza | Lift |
|---|-------|---------|-----------|------|
| 1 | Ocup: Legal → Cantidad de pasos: Activo, Estrés: Medio | 11.2% | 89.4% | 5.48 |
| 2 | Ocup: Legal → Cantidad de pasos: Activo, Edad: Medio | 11.2% | 89.4% | 5.48 |
| 3 | Ocup: Legal → BMI: Normal, Presion Arterial Sistólica: Alta | 11.2% | 89.4% | 5.39 |
| 4 | Ocup: Técnica → Frecuencia cardíaca: Baja, Presion Arterial Sistólica: Elevada | 10.2% | 53.5% | 5.27 |
| 5 | Ocup: Técnica → Presion Arterial Diastólica: Elevada, Frecuencia cardíaca: Baja | 10.2% | 53.5% | 5.27 |

![Ocupación - Top Rules](/src/output/figures/association_rules/filter_by/ocupacion/top_rules.png)

![Ocupación - Quality Matrix](/src/output/figures/association_rules/filter_by/ocupacion/quality_matrix.png)

![Ocupación - Network Graph](/src/output/figures/association_rules/filter_by/ocupacion/network_graph.png)

![Ocupación - Support vs Confidence](/src/output/figures/association_rules/filter_by/ocupacion/support_vs_confidence.png)

![Ocupación - Distribution Lift](/src/output/figures/association_rules/filter_by/ocupacion/distribution_lift.png)

---

## 7. Patrones Identificados

### 7.1 Presión Arterial

Las reglas con mayor lift global involucran presión arterial sistólica y diastólica normales, reflejando su correlación fisiológica natural. Las categorías anormales (Elevada/Alta) aparecen en combinación con otros factores como edad y ocupación.

### 7.2 Calidad y Duración del Sueño

Calidad de sueño alta se asocia con duración media/alta y actividad física moderada. Calidad baja se relaciona con estrés alto y actividad física baja. La duración media del sueño (6.4-7.8h) es la más frecuente en las reglas.

### 7.3 Estrés y Estilo de Vida

- **Estrés alto + calidad de sueño baja**: patrón consistente en reglas de insomnio
- **Estrés bajo + calidad de sueño alta**: asociado a ausencia de trastornos
- **Actividad física**: aparece como modulador relevante en reglas de ambos trastornos

### 7.4 Ocupación

- **Healthcare**: asociado a reglas de apnea del sueño y presión arterial elevada
- **Legal**: asociado a ausencia de trastornos con alta confianza y lift
- **Technical**: perfil sedentario con frecuencia cardíaca baja y presión elevada

### 7.5 Trastornos del Sueño

| Trastorno | Perfil típico | Factores asociados |
|-----------|---------------|-------------------|
| Insomnio | Sobrepeso, actividad media, edad media, presión elevada | Estrés, calidad de sueño |
| Apnea | Senior, healthcare, sobrepeso, presión arterial alta | Edad, BMI, presión arterial |
| Ninguno | Joven, legal, actividad variable, estrés variable | Calidad de sueño alta |
