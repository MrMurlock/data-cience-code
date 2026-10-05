# EDA — Sleep Health and Lifestyle Dataset

## 1. Introducción

### 1.1 Objetivo

Este informe corresponde a la etapa de análisis exploratorio de datos (EDA) del dataset *Sleep Health and Lifestyle*. El objetivo es caracterizar el dataset, evaluar su calidad, estudiar las distribuciones de sus variables y analizar un conjunto seleccionado de relaciones entre ellas. Los resultados respaldan las transformaciones aplicadas al dataset y orientan las decisiones de las etapas posteriores de análisis.

El EDA no busca demostrar hipótesis causales: sirve para describir y comprender los datos. Por eso, este documento distingue claramente entre observaciones descriptivas, asociaciones estadísticas, posibles interpretaciones y decisiones de procesamiento.

### 1.2 Descripción del dataset

El dataset incluye variables sociodemográficas (`Gender`, `Age`, `Occupation`), variables asociadas al sueño (`Sleep Duration`, `Quality of Sleep`, `Sleep Disorder`), variables de estilo de vida (`Physical Activity Level`, `Stress Level`, `BMI Category`, `Daily Steps`) y variables cardiovasculares (`Blood Pressure`, `Heart Rate`). El origen y el significado de cada columna están documentados en `DATASET.md`.

Tras el preprocessing el dataset queda con **374 registros y 13 columnas**, de las cuales 9 son numéricas (incluyendo dos derivadas de la presión arterial) y 4 categóricas.

## 2. Preparación inicial

### 2.1 Carga y tipos de datos

Los datos crudos se cargan desde `dataset/raw/original.csv` y se describe su estructura con un perfilado automático (ver `output/eda/profiling/original_profiling.html`). La tabla muestra el estado del dataset después del preprocessing:

{{ table("dataset_overview") }}

### 2.2 Transformaciones iniciales

El preprocessing aplica cuatro transformaciones reproducibles (`pipelines/eda/preprocessing.py`):

* **Separación de `Blood Pressure`** en las variables numéricas `bp_systolic` y `bp_diastolic`. La presión arterial llega como texto `"sistólica/diastólica"`; para poder analizarla numéricamente debe representarse como dos variables. La columna original se conserva solo como referencia descriptiva y no se usa como variable numérica en correlaciones ni en pruebas estadísticas.
* **Imputación de `Sleep Disorder`** con la categoría explícita de ausencia de trastorno (`No disorder`). Sin esta categoría el dataset no registraría a las personas sanas, lo que sesgaría todo el análisis (ver §2.3).
* **Estandarización de `BMI Category`**: las etiquetas `Normal Weight` y `Normal` describen el mismo estado y se unifican en `Normal`, evitando categorías duplicadas que fragmenten las frecuencias.
* **Eliminación de `Person ID`**, por ser un identificador sin contenido analítico.

Las variables derivadas tipo `age_group` o `quality_group` no se crean en este EDA: son feature engineering y solo tendrían sentido si se justificara su utilidad para un análisis posterior.

### 2.3 Tratamiento de valores faltantes

Antes de imputar, el único faltante del dataset crudo corresponde a `Sleep Disorder`:

{{ table("missing_values_raw") }}

Este faltante es aparente: en el archivo original la ausencia de trastorno se escribe como `None` (texto), que la carga en pandas interpreta como dato faltante. En realidad no hay valores perdidos por recolección fallida; lo que falta es la categoría de referencia de las personas sin trastorno. El preprocessing imputa `No disorder`, haciendo visible el grupo sano en las comparaciones de §5.2 y §5.3.

## 3. Calidad de los datos

### 3.1 Valores faltantes

Tras el preprocessing el dataset no presenta valores faltantes:

{{ table("missing_values") }}

Este resultado era esperable, dado que el único faltante (el de `Sleep Disorder`) se resuelve con la imputación de §2.3.

### 3.2 Duplicados

En el dataset limpio existen registros exactamente duplicados:

{{ metric("duplicate_stats") }}

Un registro duplicado aquí significa que comparte **todas las variables restantes** con otro registro, porque `Person ID` fue eliminado. Dado que el dataset es sintético, la explicación más plausible es que representa perfiles repetidos (personas distintas con atributos idénticos), no errores de carga de la misma persona. Por ese motivo **no se eliminan**: hacerlo alteraría artificialmente la composición de la población, por ejemplo el peso relativo de las ocupaciones y de las categorías de IMC.

Es importante notar que, si estas filas representan personas distintas, cada una aporta una observación legítima; si representaran errores del mismo individuo, se estaría duplicando información. En ambos casos, las pruebas de §5 asumen observaciones independientes, y este punto debe recordarse al interpretar los p-valores.

### 3.3 Valores inválidos y consistencia de categorías

La cardinalidad de las variables categóricas es consistente con el diccionario de datos: `Gender` tiene 2 categorías, `Occupation` 11, `BMI Category` 3 (tras la unificación `Normal Weight` → `Normal`) y `Sleep Disorder` 3 (tras la imputación).

{{ table("categorical_cardinality") }}

Los rangos numéricos (§3.4) no muestran valores fisiológicamente imposibles: la presión sistólica va de 115 a 142 mmHg y la diastólica de 75 a 95 mmHg, y la frecuencia cardíaca de 65 a 86 bpm. No hay edades negativas ni duraciones de sueño fuera de límites razonables.

Un punto de atención es `Occupation`: contiene categorías con representación mínima (`Manager` con 1 registro y `Sales Representative` con 2), que se revisan al interpretar las comparaciones por ocupación en §5.2.

### 3.4 Valores extremos y outliers

Para detectar outliers se usa la regla clásica del rango intercuartil (IQR): un valor se considera potencialmente extremo si cae fuera de `Q1 − 1.5·IQR` o `Q3 + 1.5·IQR`. En un boxplot, la caja cubre el 50% central de los datos (de Q1 a Q3), las líneas ("bigotes") se extienden hasta la observación más extrema dentro de esas vallas, y los puntos más allá se marcan individualmente como outliers.

{{ table("numeric_ranges") }}

La única variable con outliers por IQR es `Heart Rate`, con 15 registros por encima de la valla superior (78 bpm). Su distribución en §4.1 muestra por qué: la mayoría de los valores se concentra alrededor de 65–76 bpm, con una cola de valores altos de hasta 86 bpm. Al revisar los grupos de §5.2, parte de estos valores pertenece a la categoría `Sales Representative` (n=2, media de 85 bpm), muy desviada de las demás ocupaciones.

Conforme a la especificación, los outliers **no se eliminan**: se mantienen y se analizan porque corresponden a valores fisiológicamente posibles y su presencia es informativa sobre relaciones como `Sleep Duration` × `Heart Rate` (§5.1).

## 4. Análisis univariado

### 4.1 Variables numéricas

Para describir cada variable numérica se usan: tendencia central (media, mediana), dispersión (sd, cuartiles) y forma de la distribución. Dos métricas de forma ayudan a leer la tabla siguiente:

* **Asimetría (skewness)** mide si una cola de la distribución es más larga que la otra: skew > 0 indica cola hacia valores altos; skew < 0, hacia valores bajos.
* **Curtosis** mide el peso de las colas respecto de una normal: curtosis alta indica más casos extremos de lo esperado.

{{ table("numeric_summary") }}

Observaciones de la tabla, respaldadas por los gráficos:

* **`Heart Rate` es la variable con forma más problemática**: skew = 1.23 (cola derecha marcada) y curtosis = 2.29 (colas pesadas, con outliers visibles en el histograma). Es consistente con los outliers detectados en §3.4.
* **`Sleep Duration`, `Physical Activity Level`, `Stress Level` y `Quality of Sleep`** presentan asimetrías cercanas a cero, pero su curtosis negativa refleja que toman pocos valores distintos: son escalas discretas (`Quality of Sleep` y `Stress Level` están codificadas de 1 a 10, y la actividad física toma valores de 5 en 5 minutos). Deben leerse como variables cuantitativas con soporte discreto, no como normales estrictas.
* **`Age`** es aproximadamente simétrica (skew = 0.26) y abarca 27–59 años.
* **`bp_systolic` y `bp_diastolic`** son simétricas y con buena dispersión; el boxplot de la sistólica muestra un pequeño grupo de observaciones altas, dentro de un rango plausible (115–142 mmHg).

{{ image("sleep_duration_hist") }}

{{ image("heart_rate_distribution") }}

{{ image("physical_activity_hist") }}

{{ image("stress_level_hist") }}

{{ image("quality_of_sleep_hist") }}

{{ image("bp_systolic_boxplot") }}

{{ image("bp_diastolic_boxplot") }}

### 4.2 Variables categóricas

La tabla de frecuencias muestra cada categoría con su conteo absoluto y porcentual:

{{ table("categorical_frequencies") }}

Observaciones:

* **`Gender` es prácticamente equilibrada** (Male 50.5% vs Female 49.5%), lo que permite comparaciones por género sin ajuste por tamaño de grupo.
* **`Occupation` está concentrada** en unas pocas profesiones: `Nurse` (19.5%), `Doctor` (19.0%), `Engineer` (16.8%), `Lawyer` (12.6%), `Teacher` (10.7%), `Accountant` (9.9%) y `Salesperson` (8.6%) concentran alrededor del 97% de los casos, con categorías marginales (`Manager` n=1, `Sales Representative` n=2).
* **`Sleep Disorder` está moderadamente desbalanceada**: 58.6% no declara trastorno, 20.9% apnea y 20.6% insomnio. Los grupos con trastorno tienen un tamaño comparable entre sí, suficiente para las comparaciones posteriores.
* **`BMI Category` está desbalanceada**: `Normal` 57.8%, `Overweight` 39.6% y `Obese` 2.7% (10 casos). Cualquier conclusión que involucre a `Obese` debe tomarse con precaución.

{{ image("gender_distribution") }}

{{ image("occupation_distribution") }}

{{ image("sleep_disorder_distribution") }}

{{ image("bmi_category_distribution") }}

## 5. Análisis bivariado

Este análisis sigue el flujo de la especificación: primero una exploración visual, luego la cuantificación estadística de las relaciones que resultan relevantes, con métodos seleccionados según los tipos de variables involucradas. Se trata de análisis observacionales, con dos consideraciones transversales:

* Todos los test son bilaterales, con nivel de significancia α = 0.05.
* Se ejecutan varios test exploratorios en un mismo informe, sin control de comparaciones múltiples: los p-valores se leen en conjunto con el tamaño del efecto y el gráfico, no como pruebas confirmatorias aisladas.

### 5.1 Numérica–numérica

Para pares de variables numéricas se usan:

* **Correlación de Pearson (r)**: mide la fuerza de una **relación lineal**, entre −1 (lineal perfecta negativa) y +1 (lineal perfecta positiva). Es sensible a outliers y a no-linealidades.
* **Correlación de Spearman (rho)**: mide si dos variables crecen o decrecen de forma consistentemente **monotónica**, sin exigir linealidad. Se calcula sobre rangos y es más robusta a outliers.

Como heurística de lectura (orientativa): |r| < 0.25 débil, 0.25–0.5 moderada, 0.5–0.7 importante y > 0.7 fuerte.

Los resultados para los pares seleccionados:

{{ table("bivariate_numeric") }}

La matriz completa de las variables numéricas permite ver además relaciones no seleccionadas que enriquecen la lectura (por ejemplo, `Physical Activity Level` × `Daily Steps` = 0.77):

{{ table("correlation_matrix") }}

* **`Sleep Duration` × `Quality of Sleep` (r = 0.88)**: relación lineal muy fuerte: a mayor duración dormida, mayor calidad notificada. Es la relación más fuerte del bloque de sueño; la cercanía a la colinealidad de estas dos variables debe recordarse al construir modelos.

{{ image("sleep_duration_quality") }}

* **`Sleep Duration` × `Stress Level` (r ≈ −0.81)**: relación lineal fuerte pero **inversa**: menos horas de sueño se acompañan de más estrés. El patrón es claro a pesar de la discretización de ambas escalas.

{{ image("sleep_duration_stress") }}

* **`Sleep Duration` × `Heart Rate` (r = −0.52 Pearson, rho = −0.61 Spearman)**: asociación inversa entre dormir más y tener menor frecuencia cardíaca en reposo. La diferencia entre Pearson y Spearman es la señal clave (§3.4): la asociación es monotónica ya algo más estrecha de lo que sugiere Pearson, porque varios outliers de frecuencia cardíaca (personas en 85–86 bpm con sueño corto) degradan el coeficiente lineal.

{{ image("sleep_duration_heart_rate") }}

* **`Sleep Duration` × `Physical Activity Level` (r = 0.21)**: asociación positiva pero débil, con mucha dispersión: dormir más se acompaña levemente de más actividad física. Queda como asociación de fondo, no como determinante.

{{ image("sleep_duration_activity") }}

* **`bp_systolic` × `bp_diastolic` (r = 0.97)**: relación casi lineal perfecta y esperable, ya que ambas provienen de la misma medición. Se tratan como par colineal y en §6.5 se discuten las consecuencias de su uso conjunto.

{{ image("bp_scatter") }}

* **Vista global**: la matriz de correlación muestra un patrón adicional coherente con el dominio: `Quality of Sleep` está aún más correlacionada (inversamente) con `Stress Level` (−0.90) que con la duración, y `Physical Activity Level` va de la mano de `Daily Steps` (0.77). Son señales de redundancia que se retoman en §6.5.

{{ image("corr_heatmap") }}

* **Contraste Pearson vs Spearman**: repetir la misma matriz con correlación de Spearman permite aislar el efecto de los outliers. La mayoría de los pares se mantienen casi idénticos, pero **todas las relaciones que involucran `Heart Rate` quedan subestimadas por Pearson** (sus outliers estiran la nube de puntos):

  * `Sleep Duration` × `Heart Rate`: −0.52 (Pearson) → −0.61 (Spearman).
  * `Stress Level` × `Heart Rate`: 0.67 → 0.82, la mayor diferencia de toda la matriz.
  * `Quality of Sleep` × `Heart Rate`: −0.66 → −0.74.

  Un par con rho notablemente mayor que r sugiere una relación monotónica más estrecha de lo que mide el coeficiente lineal, típicamente por outliers o por curvatura. Estos pares quedan señalados como candidatos a exploración adicional (transformaciones o métodos por rangos) en los análisis posteriores.

{{ image("corr_heatmap_spearman") }}

### 5.2 Categórica–numérica

Cuando una variable numérica se divide en grupos por una categórica, la pregunta es si las distribuciones de los grupos difieren. El esquema de decisión aplicado fue:

* **2 grupos**: si los grupos son aproximadamente normales → **t de Welch** (no exige igualdad de varianzas); si no → **Mann-Whitney** (sobre rangos).
* **Más de 2 grupos**: si todos los grupos son aproximadamente normales → **ANOVA**; si no → **Kruskal-Wallis** (análogo no paramétrico del ANOVA).

La normalidad se evalúa con Shapiro-Wilk por grupo (p > 0.05). Los test no paramétricos se acompañan de tamaños de efecto sobre rangos, para distinguir diferencias grandes de diferencias significativas pero pequeñas.

Términos de lectura de la tabla:

* **p-value**: probabilidad de observar datos como los obtenidos si, en la población, los grupos no fueran distintos. Con p < 0.05 la diferencia es "estadísticamente detectable"; esto no mide su fuerza ni su importancia.
* **Cohen's d** (t de Welch): diferencia de medias en desviaciones estándar: 0.2 pequeña, 0.5 moderada, 0.8 grande.
* **r rank-biserial** (Mann-Whitney): dominancia de un grupo sobre otro basada en rangos: ~0.1 pequeña, ~0.3 moderada, ~0.5 grande.
* **eta²** (ANOVA): proporción de varianza explicada por la categoría: 0.01 pequeña, 0.06 moderada, 0.14 grande.
* **epsilon²** (Kruskal-Wallis): análogo de eta² sobre rangos: 0.01 pequeña, 0.08 moderada, 0.26 grande.

{{ table("group_tests") }}

Todas las comparaciones fallaron el supuesto de normalidad por grupo (Shapiro); parte de la razón es que las variables de sueño y estrés son escalas discretas con ties masivos. De ahí la preferencia sistemática por test de rangos y sus tamaños de efecto.

Las relaciones destacadas:

* **`Sleep Disorder` × `Quality of Sleep` (epsilon² = 0.13, p ≈ 2·10⁻¹¹)**: la calidad notificada se ordena claramente por grupo: `Insomnia` (media 6.5) por debajo de `Sleep Apnea` (7.2) y `No disorder` (7.6). La diferencia apunta sobre todo a un insomnio con peor calidad percibida.

{{ image("disorder_quality_of_sleep_boxplot") }}

* **`Sleep Disorder` × `Sleep Duration` (epsilon² = 0.12, p ≈ 3·10⁻¹⁰)**: las personas con insomnio duermen menos (media 6.59 h) que sin trastorno (7.36 h) y con apnea (7.03 h).

{{ image("disorder_sleep_duration_boxplot") }}

* **`Sleep Disorder` × `Heart Rate` (epsilon² = 0.10, p ≈ 8·10⁻⁹)**: el grupo con apnea presenta la mayor frecuencia cardíaca (media 73.1 bpm), seguido de insomnio (70.5) y del grupo sin trastorno (69.0). Es coherente con la fisiología de la apnea y coincide con la señal de outliers de §3.4.

{{ image("disorder_heart_rate_boxplot") }}

* **`Sleep Disorder` × `Stress Level` (epsilon² = 0.03, p = 0.007)**: comparación significativa pero de **efecto pequeño**: el estrés de quienes declaran trastorno solo es levemente mayor (5.9 insomnio y 5.7 apnea vs 5.1 sin trastorno). Es un caso instructivo de p-valor pequeño con asociación débil.

{{ image("disorder_stress_level_boxplot") }}

* **`BMI Category` × `Heart Rate` (epsilon² = 0.14, p ≈ 2·10⁻¹²)**: el contraste más grande entre grupos de IMC: `Obese` tiene media de 84.3 bpm frente a 69.0 (`Normal`) y 71.0 (`Overweight`). Aunque el grupo `Obese` tiene solo 10 casos, la diferencia es enorme; debe recordarse que el tamaño del grupo limita la precisión de la estimación.

{{ image("bmi_heart_rate_boxplot") }}

* **`BMI Category` × `Sleep Duration` (epsilon² = 0.12)** y **`BMI Category` × `Quality of Sleep` (epsilon² = 0.10)**: diferencias en la dirección esperada (menos horas y peor calidad en `Overweight`/`Obese`), con niveles intermedios de calidad de sueño (6.9 y 6.4 respectivamente) contra 7.6 de `Normal`.

{{ image("bmi_sleep_duration_boxplot") }}

* **`Gender` × `Stress Level`, `Quality of Sleep` y `Heart Rate` (r rank-biserial 0.34–0.48, p < 10⁻⁸)**: los hombres reportan más estrés (media 6.08 vs 4.68), menor calidad de sueño (6.97 vs 7.66) y mayor frecuencia cardíaca (71.05 vs 69.26). Con grupos de tamaño equivalente, son diferencias de fondo; la mayor es la del estrés.

{{ image("gender_quality_of_sleep_boxplot") }}

* **`Occupation` × variables de sueño**: Kruskal-Wallis muestra asociaciones grandes (epsilon² 0.34–0.45, p < 10⁻²²): la calidad de sueño va de 8.4 en `Engineer` a 4.0 en `Sales Representative` y 6.0 en `Salesperson`; la frecuencia cardíaca va de 67.2 en `Engineer` a 85.0 en `Sales Representative`. La advertencia es importante: los tamaños mínimos de grupo son 1 (`Manager`) y 2 (`Sales Representative`), responsables de buena parte de la señal de `Heart Rate`. Esta comparación **describe muy bien el dataset**, pero cualquier conformación formal debería reagrupar o excluir los niveles marginales.

{{ image("occupation_sleep_duration_boxplot") }}

### 5.3 Categórica–categórica

Cuando dos variables son categóricas, el análisis se basa en:

* **Tabla de contingencia**: conteo de observaciones por combinación de categorías.
* **Chi-cuadrado de independencia (χ²)**: compara los conteos observados con los **conteos esperados** que tendría cada celda si las variables fueran independientes. Requiere conteos esperados suficientes (regla práctica: ninguna celda bajo 1 y no más del 20% de celdas bajo 5); con celdas esperadas muy pequeñas, χ² se infla y puede dar significancias espurias.
* **V de Cramér**: tamaño de efecto derivado de χ², normalizado entre 0 (sin asociación) y 1 (asociación perfecta): ~0.1 pequeña, ~0.3 moderada, ~0.5 fuerte (lectura orientativa).

{{ table("categorical_associations") }}

Las tres asociaciones son estadísticamente detectables y con efecto material:

* **`Occupation` × `Sleep Disorder` (V = 0.75, p ≈ 7·10⁻⁷⁷)**: la asociación más fuerte del dataset. La contingencia muestra patrones extremos: de 32 `Salesperson`, 29 tienen insomnio; de 73 `Nurse`, 61 tienen apnea. Como ambas son ocupaciones numerosas (§4.2), la señal no es un artefacto del tamaño; sin embargo, **36.4% de las celdas tienen conteo esperado < 5** (mínimo 0.21), por lo que χ² sobreestima la confianza. La asociación es fuerte pero su p-valor debe tomarse como aproximado.

{{ table("contingency_occupation_sleep_disorder") }}

{{ image("occupation_sleep_disorder_prop") }}

* **`BMI Category` × `Sleep Disorder` (V = 0.57, p ≈ 6·10⁻⁵²)**: el grueso del grupo `Normal` no declara trastorno (200 de 216), mientras que el grupo `Obese` nunca declara "No disorder": 4 con insomnio y 6 con apnea. La interpretación está limitada porque 22.2% de las celdas esperadas queda bajo 5 (mínimo 2.06), en parte por los 10 casos de `Obese`.

{{ table("contingency_bmi_sleep_disorder") }}

{{ image("bmi_sleep_disorder_prop") }}

* **`Gender` × `Sleep Disorder` (V = 0.38, p ≈ 2·10⁻¹²)**: sin problemas de supuestos (todas las celdas esperadas sobre 38). El patrón notable es el de la apnea casi exclusivamente femenina dentro del dataset: 67 de los 78 casos de apnea corresponden a mujeres, mientras que solo 11 de 189 hombres la presentan. Es un contraste descriptivo que merece una mirada clínica, no una inferencia causal.

{{ table("contingency_gender_sleep_disorder") }}

{{ image("gender_sleep_disorder_prop") }}

## 6. Principales hallazgos

### 6.1 Relaciones relevantes

* `Sleep Duration` × `Quality of Sleep`: la asociación más fuerte del bloque de sueño: {{ metric("sleep_duration_quality_pearson") }}.
* `Sleep Duration` × `Stress Level`: asociación fuerte inversa: {{ metric("sleep_duration_stress_pearson") }}.
* `Sleep Duration` × `Heart Rate`: asociación moderada inversa, con Spearman más informativa que Pearson por los outliers: {{ metric("sleep_duration_heart_rate_pearson") }}, {{ metric("sleep_duration_heart_rate_spearman") }}.
* `Sleep Duration` × `Physical Activity Level`: asociación positiva débil: {{ metric("sleep_duration_activity_pearson") }}.
* `bp_systolic` × `bp_diastolic`: identidad funcional: {{ metric("bp_scatter_pearson") }}.
* Diferencias de grupo: el insomnio duerme menos y con peor calidad; la apnea tiene mayor frecuencia cardíaca; `Obese` tiene una frecuencia cardíaca muy superior (media 84.3 bpm); las ocupaciones separan fuertemente el perfil de sueño (epsilon² 0.34–0.45).
* Asociaciones categóricas: ocupación–trastorno (V = 0.75), IMC–trastorno (V = 0.57) y género–trastorno (V = 0.38). Los patrones más contundentes son `Salesperson` → insomnio, `Nurse` → apnea y apnea predominantemente femenina.

### 6.2 Distribuciones relevantes

* `Heart Rate` es la única variable numérica con asimetría y curtosis anómalas (skew 1.23, curtosis 2.29) y la única con outliers por IQR.
* `Sleep Duration`, `Physical Activity Level`, `Stress Level` y `Quality of Sleep` son simétricas pero discretas, con curtosis negativa que refleja pocos valores distintos.
* `Age` es casi simétrica y con rango amplio (27–59 años).
* Las variables derivadas de presión arterial son simétricas y plausibles.

### 6.3 Outliers

* Solo `Heart Rate` los presenta (15 registros sobre la valla superior de 78 bpm). Se conservan porque son valores fisiológicamente posibles y son explicativos de la asociación inversa con la duración del sueño.
* Parte de esos valores se concentra en ocupaciones marginales (`Sales Representative`, n=2), lo que refuerza la advertencia de §5.2 sobre los grupos pequeños.

### 6.4 Desbalance de categorías

* `Gender` está equilibrada (~50/50).
* `Sleep Disorder` concentra 58.6% en "sin trastorno", con 20.9% de apnea y 20.6% de insomnio.
* `BMI Category` se concentra en `Normal` (57.8%) y `Overweight` (39.6%), con `Obese` reducido a 10 casos.
* `Occupation` concentra ~97% de los casos en 7 de sus 11 categorías; `Manager` y `Sales Representative` son marginales.

### 6.5 Variables potencialmente redundantes

* `bp_systolic` y `bp_diastolic` (r = 0.97) son casi duplicadas entre sí y provienen de la misma medición; no deben usarse juntas como predictores sin registrar la colinearidad.
* `Physical Activity Level` (min/día) y `Daily Steps` (r = 0.77) miden el mismo fenómeno en escalas distintas; conservar ambas solo tiene sentido si el análisis distingue intensidad de registro diario.
* `Quality of Sleep`, `Sleep Duration` y `Stress Level` (r de 0.88 y −0.90) llevan información muy solapada: al modelar una de las tres, las otras aportan poca información lineal adicional.

## 7. Decisiones de procesamiento

### 7.1 Transformaciones realizadas

* Imputación de `Sleep Disorder` con la categoría `No disorder`, que expone al grupo sano.
* Estandarización de `BMI Category` (`Normal Weight` → `Normal`), por semántica y frecuencia.
* Separación de `Blood Pressure` en `bp_systolic` y `bp_diastolic` para habilitar el análisis numérico.

### 7.2 Variables eliminadas

* `Person ID`: identificador sin uso analítico. Su eliminación explica los registros duplicados de §3.2, que se decidió conservar.
* No se eliminó ninguna otra variable: los outliers se analizan y se conservan, y no se fusionaron categorías más allá de la estandarización de IMC.

### 7.3 Consideraciones para análisis posteriores

* **Correlación no implica causalidad.** Todas las relaciones de §5 son observacionales: no hay asignación experimental ni control de confusores.
* **Los duplicados (64.7%) y las escalas discretas deben informar el diseño de modelos**: por ejemplo, al validar un modelo conviene agrupar perfiles idénticos en el mismo lado del split (evitar fuga de datos) y estratificar por ocupación.
* **Los grupos pequeños limitan la inferencia**: `Obese` (n=10), `Manager` (n=1), `Sales Representative` (n=2). Los análisis futuros deberían reagruparlos o excluirlos/justificar su tratamiento explícitamente.
* **Como siguiente paso natural** se sugiere análisis multivariado: por ejemplo, examinar si la asociación de la apnea con la presión arterial persiste tras controlar por edad, dado que `Quality of Sleep` y `Stress Level` están casi colineales (r = −0.90).
* **Para pruebas paramétricas futuras**: tratar `Quality of Sleep` y `Stress Level` como ordinales y usar métodos por rangos como línea base, dado el patrón de ties masivos.

## 8. Conclusiones

El EDA entrega una caracterización clara y reproducible del dataset. En el plano descriptivo, el dataset es pequeño (374 perfiles), completo tras la imputación de `Sleep Disorder`, sin valores imposibles y con un único foco de outliers (`Heart Rate`). En el plano asociativo se destacan tres ejes: la buena calidad de sueño se acompaña de más horas de sueño y menos estrés; los grupos con trastorno del sueño difieren de manera detectable en duración, calidad y frecuencia cardíaca; y ocupación, IMC y género están asociados de forma marcada a la presencia de trastornos del sueño.

Estas son asociaciones estadísticas sobre datos observacionales, no mecanismos causales. El análisis es exploratorio y los p-valores son aproximados por comparación múltiple y por los duplicados del dataset; sirve para priorizar hipótesis y decisiones de preprocessing, no para confirmar teorías sobre el sueño.

Como resultado de la etapa, el dataset procesado en `dataset/processed/clean.csv` queda listo para análisis posteriores con las transformaciones de §7. Este informe y su presentación se generan a partir de los mismos artifacts en `artifacts/eda/`, de modo que toda cifra se puede actualizar volviendo a ejecutar el pipeline y renderizando nuevamente los documentos.
