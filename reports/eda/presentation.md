---
marp: true
theme: default
paginate: true
---

# EDA — Sleep Health and Lifestyle Dataset

Análisis exploratorio para la ayudantía de ciencia de datos

---

## ¿Qué es el EDA?

**Exploratory Data Analysis**: mirar los datos *antes* de modelar.

* Caracterizar: ¿cuántos registros, qué variables, de qué tipo?
* Evaluar calidad: faltantes, duplicados, outliers.
* Describir: distribuciones de cada variable.
* Explorar: relaciones entre variables relevantes.

> Regla de la ayudantía: el EDA *describe y sugiere*, no confirma causalidad.

---

## Dataset

* 374 registros × 13 columnas (tras preprocessing).
* Sueño: `Sleep Duration`, `Quality of Sleep`, `Sleep Disorder`.
* Estilo de vida: actividad física, estrés, IMC, pasos.
* Cardiovascular: presión arterial (sistólica/diastólica), `Heart Rate`.

---

## Preparación: 4 transformaciones

| Transformación | Por qué |
|---|---|
| `Blood Pressure` → `bp_systolic` + `bp_diastolic` | viene como texto `sist/diast`; se necesita numérica |
| `Sleep Disorder` missing → `No disorder` | `None` era la categoría "sano", no un dato perdido |
| `Normal Weight` → `Normal` | dos etiquetas para el mismo estado |
| Eliminar `Person ID` | identificador sin valor analítico |

---

## ¿Cómo se leen los faltantes?

{{ table("missing_values_raw") }}

El único faltante del dataset crudo es *aparente*: `Sleep Disorder` vacío **significa** "sin trastorno".

Tras imputar, el dataset queda 100% completo.

---

## Calidad: duplicados

{{ metric("duplicate_stats") }}

Registros idénticos **porque quitamos `Person ID`**: perfiles repetidos del dataset sintético.

Decisión: mantenerlos (quitarlos cambiaría la composición de la población). La interpretación de p-valores debe recordar este punto.

---

## Univariado numérico: `Heart Rate`

La única variable con forma no uniforme:

* **Skew = 1.23** → cola derecha marcada.
* **Curtosis = 2.29** → colas pesadas, con casos extremos.
* Media ≈ 70 bpm, rango 65–86 → **15 outliers** sobre 78 bpm (regla IQR).

{{ image("heart_rate_distribution", 460) }}

---

## Univariado numérico: el resto es casi uniforme

* `Sleep Duration`, `Physical Activity`, `Stress`, `Quality of Sleep`: **skew ≈ 0** y **curtosis negativa** → pocos valores distintos, escalas discretas, sin colas.
* `Age`: casi simétrica (27–59 años); presión arterial simétrica y plausible (115–142 / 75–95 mmHg).

{{ image("sleep_duration_hist", 330) }} {{ image("quality_of_sleep_hist", 330) }}

---

## Univariado categórico: `Gender`

Prácticamente equilibrada: **Male 189 (50.5%) vs Female 185 (49.5%)**.

Compara grupos de casi el mismo tamaño: las diferencias de medias no se explican por el desbalance de muestras.

{{ image("gender_distribution", 380) }}

---

## Univariado categórico: `Occupation` muy concentrada

* 7 profesiones concentran ~97% de los casos (`Nurse` 19.5%, `Doctor` 19.0%, `Engineer` 16.8%, …).
* **Categorías marginales**: `Manager` (n=1), `Sales Representative` (n=2), `Scientist` (n=4).

Consecuencia: cualquier comparación formal por ocupación necesita reagrupar o excluir esos niveles.

{{ image("occupation_distribution", 420) }}

---

## Univariado categórico: `BMI` y `Sleep Disorder`

* `BMI`: `Normal` 57.8% / `Overweight` 39.6% / `Obese` **solo 10 casos (2.7%)** → cuidado al concluir sobre `Obese`.
* `Sleep Disorder`: "sin trastorno" 58.6%; apnea 20.9% e insomnio 20.6% (grupos suficientes para comparar).

{{ image("bmi_category_distribution", 380) }} {{ image("sleep_disorder_distribution", 380) }}

---

## Análisis bivariado: ¿qué buscamos?

Tres combinaciones de tipos de variables, cada una con su método:

| Combinación | Pregunta | Herramientas |
|---|---|---|
| numérica × numérica | ¿se mueven juntas? | scatter plots + Pearson/Spearman |
| categórica × numérica | ¿difieren los grupos? | boxplots + t-test/ANOVA/MW/KW |
| categórica × categórica | ¿están asociadas? | contingencias + χ² + Cramér's V |

No se prueban todas las combinaciones: se seleccionan relaciones relevantes por dominio y por lo que sugirió el análisis visual.

---

## ¿Qué es la correlación? (numérica × numérica)

* **Pearson (r)**: fuerza de la relación **lineal** (−1 a +1). Sensible a outliers y curvatura.
* **Spearman (rho)**: relación **monotónica** calculada sobre rangos. Más robusta.

Lectura orientativa de \|r\|: < 0.25 débil · 0.25–0.5 moderada · 0.5–0.7 importante · > 0.7 fuerte.

* **Colinealidad**: dos numéricas explican casi lo mismo (|r| alto) — problema al modelar.
* **Multicolineariedad**: lo mismo pero entre tres o más numéricas.

> r alto **no** significa causalidad.

---

## Duración × Calidad de sueño: r = 0.88

{{ image("sleep_duration_quality", 470) }}

La relación más fuerte del bloque de sueño: casi **colineales** — al modelar una, la otra aporta poco.

---

## Duración × Estrés: r = −0.81

{{ image("sleep_duration_stress", 470) }}

Relación fuerte e **inversa**: dormir menos se acompaña de más estrés (a pesar de las escalas discretas).

---

## ¿Qué es un outlier y por qué importa ahora?

* Regla IQR: outlier si cae fuera de `[Q1 − 1.5·IQR, Q3 + 1.5·IQR]`.
* `Heart Rate` concentra **los únicos 15 outliers** del dataset (> 78 bpm).
* Se conservan: son plausibles y **degradan justamente los estadísticos sensibles a colas**, como la correlación de Pearson y las medias.

---

## Duración × Frec. cardíaca: los outliers explicando la diferencia

{{ image("sleep_duration_heart_rate", 470) }}

**Pearson −0.52 vs Spearman −0.61**: los puntos rojos (85–86 bpm con sueño corto) degradan el coeficiente *lineal*; Spearman (rangos) es más robusto y captura mejor la relación monotónica.

---

## Dos matrices para comparar

{{ image("corr_heatmap", 400) }} {{ image("corr_heatmap_spearman", 400) }}

---

## Lectura conjunta: dónde Pearson se queda corto

* La mayoría de los pares coinciden en ambas matrices: relaciones lineales sin distorsión.
* **Todas las relaciones con `Heart Rate` son subestimadas por Pearson** (sus outliers estiran la nube):
  * Duración × Frec. cardíaca: **−0.52 → −0.61**
  * Estrés × Frec. cardíaca: **0.67 → 0.82** (¡la mayor diferencia!)
  * Calidad × Frec. cardíaca: **−0.66 → −0.74**
* Contraste útil: pares con rho ≫ r son **candidatos a exploración adicional** (curvatura u outliers) → transformaciones o métodos por rangos en las etapas siguientes.

---

## Señales de redundancia (para modelar)

* `bp_systolic` × `bp_diastolic` = 0.97 → casi duplicadas.
* `Physical Activity` × `Daily Steps` = 0.77.
* `Quality of Sleep` × `Stress Level` = −0.90; `Sleep Duration` × `Quality` = 0.88.

Ver §6.5 del reporte para las decisiones asociadas.

---

## ¿Comparar grupos? Métodos, tipo de variable y criterios

Se compara una variable **numérica continua** *entre los grupos* de una variable **categórica**:

| Método | Nº de grupos | Exige normalidad? |
|---|---|---|
| t de Welch | 2 | sí, por grupo (Shapiro p > 0.05) |
| ANOVA | > 2 | sí, en todos los grupos |
| Mann-Whitney U | 2 | no (opera sobre rangos) |
| Kruskal-Wallis | > 2 | no (opera sobre rangos) |

Si la normalidad falla → versión no paramétrica. Reportar siempre el **tamaño de efecto**.
La normalidad se comprueba utilizando el test **Shapiro-Wilk**.

---

## ¿Qué es el p-valor?

* Es la probabilidad de observar datos como los obtenidos (o más extremos) **si no hubiera diferencia real** en la población (hipótesis nula).
* Con **p < 0.05**: los datos son incompatibles con "no hay diferencia" → diferencia *detectable*.
* **No es** la probabilidad de que "no haya diferencia" ni tampoco la de que la haya: solo califica la evidencia contra la nula.

---

## Tamaños de efecto: qué miden y cuánto importa

* **d de Cohen** *(t-test)*: distancia de medias en desviaciones estándar → 0.2 pequeña / 0.5 moderada / 0.8 grande.
* **r rank-biserial** *(Mann-Whitney)*: dominancia de un grupo sobre otro, sobre rangos → 0.1 / 0.3 / 0.5.
* **eta²** *(ANOVA)*: fracción de la varianza de la numérica explicada por la categoría → 0.01 / 0.06 / 0.14.
* **epsilon²** *(Kruskal-Wallis)*: análogo de eta² sobre rangos → 0.01 / 0.08 / 0.26.

> Un p-valor minúsculo **no** implica un efecto importante: siempre leer ambos.

---

## Trastorno del sueño × sueño y cardíaco (tabla)

| Comparación | Efecto (epsilon²) | Lectura |
|---|---|---|
| × Calidad de sueño | 0.13 | insomnio: peor calidad (media 6.5 vs 7.6, escala 1–10) |
| × Duración | 0.12 | insomnio duerme menos (6.59 vs 7.36 h/día) |
| × Frec. cardíaca | 0.10 | apnea: más bpm (73.1 vs 69.0) |
| × Estrés | **0.03** | ¡p = 0.007 pero efecto pequeño! |

---

## ¿Cómo se lee un boxplot?

* **Caja**: rango intercuartil (Q1–Q3), el 50% central de los datos.
* **Línea interna**: mediana.
* **Bigotes**: hasta el último dato dentro de 1.5·IQR.
* **Puntos**: outliers individuales.

En las siguientes láminas se usa para **comparar la numérica entre grupos** de la categórica.

---

## Trastorno × Duración y Frec. cardíaca (boxplots)

{{ image("disorder_sleep_duration_boxplot", 400) }} {{ image("disorder_heart_rate_boxplot", 400) }}

Insomnio con menos horas; apnea con mayor frecuencia cardíaca.

---

## IMC × Frecuencia cardíaca: el contraste más grande

{{ image("bmi_heart_rate_boxplot", 470) }}

`Obese` (n=10): media **84.3 bpm** vs ~69–71 en los demás grupos.

Advertencia: grupo pequeño → estimación imprecisa.

---

## Género: diferencias de fondo (medias)

Con grupos de tamaño casi idéntico (185 vs 189), se comparan **medias**:

| Variable | Male | Female | Efecto |
|---|---|---|---|
| Estrés (escala 1–10) | 6.08 | 4.68 | moderado–grande (r = 0.48) |
| Calidad de sueño (1–10) | 6.97 | 7.66 | moderado (r = 0.34) |
| Frec. cardíaca (bpm) | 71.05 | 69.26 | moderado (r = 0.35) |
| Duración (h/día) | 7.04 | 7.23 | pequeño (p = 0.014) |

---

## Género: diferencias de fondo (medias)

{{ image("gender_quality_of_sleep_boxplot", 400) }}

---

## Ocupación: señal fuerte, grupos frágiles

* Kruskal-Wallis: epsilon² 0.34–0.45; calidad de sueño desde 8.4 (`Engineer`) hasta 4.0 (`Sales Representative`), escala 1–10.
* Pero **grupos de n=1 y n=2 concentran parte de la señal** (p. ej., en frecuencia cardíaca).

{{ image("occupation_sleep_duration_boxplot", 480) }}

Describir el dataset: sí. Inferir formalmente: reagrupar niveles primero.

---

## Categórica × categórica: la idea del χ² (1/2)

Se compara el **conteo observado** de cada celda con el **conteo esperado** si las dos variables fueran independientes.

Con este dataset: si género no influyera en la apnea (20.9% del total ≈ 20.9% de hombres y de mujeres), esperaríamos ~39 casos de apnea por género. Lo observado: **67 mujeres vs 11 hombres** → la desviación es lo que mide χ².

---

## Categórica × categórica: χ² y sus supuestos (2/2)

* **χ²** acumula las discrepancias `(observado − esperado)² / esperado` de todas las celdas: valores grandes → indicios de asociación.
* **Supuesto clave**: conteos esperados suficientes (ninguno < 1; ≤ 20% de celdas < 5).
* Si falla, χ² se inflan y aparecen significancias espurias → reportar y advertir (pasa con `Occupation`).

---

## V de Cramér: de χ² a una escala comparable

* Fórmula: `V = √( χ² / ( n · (min(filas, columnas) − 1) ) )`.
* Rango **0 a 1**: 0 = sin asociación; 1 = asociación perfecta.
* Lectura orientativa: ~0.1 pequeña · ~0.3 moderada · ~0.5 fuerte.

Permite comparar la fuerza de asociaciones entre tablas de distinto tamaño.

---

## Asociaciones con trastornos del sueño

| Par | V de Cramér | Patrón |
|---|---|---|
| Occupation × Disorder | **0.75** | Salesperson → insomnio (29/32); Nurse → apnea (61/73) |
| BMI × Disorder | 0.57 | Obese nunca "No disorder" (0/10) |
| Gender × Disorder | 0.38 | Apnea: 67/78 casos en mujeres |

*Occupation* tiene 36% de celdas esperadas < 5 → p-valor aproximado, interpretar con cuidado.

---

## Principales hallazgos

* Dormir más ↔ mejor calidad (0.88) y menos estrés (−0.81).
* Las relaciones con `Heart Rate` son más fuerte en Spearman → efecto de outliers.
* Grupos con trastorno: menos sueño, peor calidad, más bpm.
* `Obese`: frecuencia cardíaca muy superior (84 bpm).
* Ocupación, IMC y género asociados fuertemente a trastornos.

---

## Decisiones de procesamiento

* Imputar `Sleep Disorder` → `No disorder` (exponer al grupo sano).
* Unificar `Normal Weight` → `Normal`.
* Separar presión arterial en dos numéricas; no usar la original.
* Eliminar `Person ID` (origen de duplicados, se conservan).
* Cuidado con grupos pequeños: `Obese` (10), `Sales Rep` (2), `Manager` (1).

---

## Conclusiones

* El dataset es pequeño pero completo y plausible tras el preprocessing.
* Tres ejes asociativos: sueño↔estrés/calidad, trastornos × perfiles, categorías × trastornos.
* Pearson vs Spearman: outliers **sí cambian la lectura** — comparar ambos es parte del método.
* Todo es **observacional**: asociaciones, no causalidad.
* Listo para la siguiente etapa: dataset limpio + artifacts reproducibles.
