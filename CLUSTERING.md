# Análisis de Clustering: Dataset de Salud del Sueño y Estilo de Vida

---

## 1. Introducción

Antes que nada, necesitamos entender qué es lo que queremos lograr con el clustering.

El clustering es una técnica de aprendizaje no supervisado que busca agrupar observaciones similares sin disponer de etiquetas previas. A diferencia de la clasificación, donde conocemos las categorías de antemano, en clustering el algoritmo debe descubrir la estructura inherente en los datos.

La pregunta fundamental que debemos responder es:

**¿Qué significa que dos registros (personas) sean similares?**

La respuesta a esta pregunta depende de dos aspectos clave:
1. **Las variables utilizadas**: ¿Consideramos similares a dos personas que tienen la misma edad, o a dos personas que tienen el mismo nivel de estrés?
2. **La distancia definida entre observaciones**: ¿Cómo medimos matemáticamente la similitud? ¿Usamos distancia euclidiana, distancia de Manhattan, o alguna otra métrica?

En este análisis, trabajaremos con el **Dataset de Salud del Sueño y Estilo de Vida**, que contiene 374 registros de individuos con información sobre:
- Edad, género, ocupación
- Duración y calidad del sueño
- Nivel de actividad física y estrés
- Presión arterial y frecuencia cardíaca
- Pasos diarios y trastornos del sueño

El objetivo general del análisis es descubrir grupos naturales de individuos con patrones similares de sueño y salud, lo que podría ayudar a entender mejor los factores que influyen en la calidad del sueño y identificar segmentos de población con características distintivas.

---

## 2. Preparación de los datos

Para realizar el clustering, es necesario preparar adecuadamente los datos. En este análisis:

**Variables utilizadas**: Se utilizaron únicamente variables cuantitativas. Las variables categóricas (Gender, Occupation, BMI Category, Sleep Disorder, age_group, quality_group) fueron excluidas para esta etapa del análisis.

Las variables cuantitativas incluidas fueron:
- Age
- Sleep Duration
- Quality of Sleep
- Physical Activity Level
- Stress Level
- Heart Rate
- Daily Steps
- bp_systolic (presión arterial sistólica)
- bp_diastolic (presión arterial diastólica)

**Normalización de datos**: Las variables cuantitativas fueron normalizadas utilizando StandardScaler (estandarización), que transforma cada variable para que tenga media 0 y desviación estándar 1.

¿Por qué es importante la normalización?
- **Para PCA**: PCA busca direcciones de máxima varianza. Si una variable tiene una escala mucho mayor que las demás (por ejemplo, Daily Steps con valores de 3000-10000 vs Stress Level con valores de 3-8), dominará los componentes principales simplemente por su escala, no por su importancia real.
- **Para K-Means**: K-Means minimiza la suma de distancias cuadradas al centroide. Sin normalización, las variables con mayor escala tendrán un peso desproporcionado en el cálculo de la distancia.
- **Para clustering jerárquico**: La distancia entre observaciones se ve afectada por las escalas de las variables. La normalización asegura que todas las variables contribuyan equitativamente.
- **Para DBSCAN**: DBSCAN define densidad basándose en la distancia entre puntos. Sin normalización, las variables con mayor escala dominarían el cálculo de distancia.

---

## 3. Reducción de dimensionalidad mediante PCA

### ¿Qué es PCA?

PCA (Principal Component Analysis) es una técnica de reducción de dimensionalidad que transforma las variables originales en un nuevo conjunto de variables llamadas **componentes principales**. Estos componentes son combinaciones lineales de las variables originales y tienen dos propiedades importantes:
1. Son ortogonales entre sí (no correlacionados)
2. Capturan la máxima varianza posible en orden descendente (el primer componente captura más varianza que el segundo, el segundo más que el tercero, etc.)

### Resultados del PCA

Se calcularon componentes principales a partir de las 9 variables normalizadas. A continuación se presenta la tabla de varianza explicada:

| Componente | Varianza Explicada | Varianza Acumulada |
|------------|-------------------|-------------------|
| PC1 | 38.94% | 38.94% |
| PC2 | 30.97% | 69.91% |
| PC3 | 17.23% | 87.14% |
| PC4 | 6.71% | 93.85% |
| PC5 | 2.97% | 96.82% |
| PC6 | 1.50% | 98.32% |
| PC7 | 0.89% | 99.21% |
| PC8 | 0.64% | 99.85% |
| PC9 | 0.15% | 100.00% |

### Decisión sobre el número de componentes

Se evaluaron representaciones con 2 y 3 componentes principales:
- **2 componentes**: Capturan el 69.91% de la varianza total. Permite visualización en 2D pero pierde información importante.
- **3 componentes**: Capturan el 87.14% de la varianza total. Permite visualización en 3D y retiene la mayoría de la información.

Finalmente se decidió utilizar **3 componentes principales** para el análisis posterior, ya que:
- Capturan más del 87% de la varianza total
- Permiten visualización en 3D, que es más intuitiva para interpretar clusters
- La ganancia de varianza adicional al agregar más componentes es marginal

### Interpretación de PCA

**Componentes principales**: Son las nuevas variables creadas por PCA. Cada componente es una combinación lineal de las variables originales.

**Scores**: Son los valores de cada observación en el espacio de los componentes principales. Por ejemplo, el score PC1 de una persona indica su posición a lo largo del primer componente principal.

**Loadings**: Son los coeficientes que indican cuánto contribuye cada variable original a cada componente principal. Los loadings nos permiten interpretar qué representan los componentes.

A continuación se presentan los loadings para los primeros 3 componentes:

| Variable | PC1 | PC2 | PC3 |
|----------|-----|-----|-----|
| Age | 0.2399 | 0.4186 | -0.2943 |
| Sleep Duration | 0.4809 | 0.0172 | 0.0760 |
| Quality of Sleep | 0.5163 | 0.0585 | 0.0425 |
| Physical Activity Level | 0.0468 | 0.3772 | 0.5698 |
| Stress Level | -0.5022 | -0.0136 | 0.1120 |
| Heart Rate | -0.4110 | 0.1109 | -0.0238 |
| Daily Steps | -0.0303 | 0.2793 | 0.6575 |
| bp_systolic | -0.1046 | 0.5283 | -0.3083 |
| bp_diastolic | -0.0972 | 0.5560 | -0.2017 |

**Interpretación de los componentes**:
- **PC1 (38.94%)**: Tiene loadings positivos altos en Sleep Duration (0.48) y Quality of Sleep (0.52), y loadings negativos en Stress Level (-0.50) y Heart Rate (-0.41). Este componente parece representar la **calidad general del sueño y el bienestar**: valores altos indican mejor sueño y menor estrés/frecuencia cardíaca.
- **PC2 (30.97%)**: Tiene loadings positivos altos en bp_diastolic (0.56), bp_systolic (0.53) y Age (0.42). Este componente parece representar la **presión arterial y la edad**: valores altos indican mayor presión arterial y edad más avanzada.
- **PC3 (17.23%)**: Tiene loadings positivos altos en Daily Steps (0.66) y Physical Activity Level (0.57). Este componente parece representar el **nivel de actividad física**: valores altos indican mayor actividad física.

### Visualizaciones PCA

![Biplot 2D](/src/output/figures/pca/biplot_2d_pca.png)

![Biplot 3D](/src/output/figures/pca/biplot_3d_pca.png)

**Cómo interpretar los biplots**:
- Los puntos representan las observaciones (personas) en el espacio de los componentes principales.
- Los vectores (flechas) representan las variables originales.
- La **longitud** de un vector indica cuánto contribuye esa variable al componente (loading más alto = vector más largo).
- La **dirección** de un vector indica la correlación con el componente: vectores que apuntan en la misma dirección están correlacionados positivamente, vectores que apuntan en direcciones opuestas están correlacionados negativamente.
- La **posición de un punto** relativa a un vector indica el valor de esa variable para esa observación: puntos cerca de la dirección de un vector tienen valores altos en esa variable.

---

## 4. Clustering mediante K-Means

### ¿Cómo funciona K-Means?

K-Means es un algoritmo de particionamiento que divide los datos en K clusters. El algoritmo funciona de la siguiente manera:
1. Inicializa K centroides de manera aleatoria
2. Asigna cada observación al centroide más cercano
3. Recalcula los centroides como el promedio de las observaciones en cada cluster
4. Repite los pasos 2 y 3 hasta que los centroides no cambian significativamente

K-Means minimiza la **inertia**, que es la suma de las distancias cuadradas de cada observación a su centroide asignado.

### Selección del número óptimo de clusters (K)

Para determinar el valor óptimo de K, se utilizaron dos métodos complementarios:

#### Método del codo (Elbow Method)

El método del codo evalúa cómo disminuye la inertia a medida que aumenta K. La idea es encontrar el "codo" de la curva, donde la ganancia marginal de agregar otro cluster deja de ser significativa.

![Método del Codo](/src/output/figures/clustering/kmeans/elbow_method.png)

Los valores de inertia obtenidos fueron:
- K=2: 1933.58
- K=3: 1422.69
- K=4: 876.29
- K=5: 643.26
- K=6: 455.95
- K=7: 314.05
- K=8: 249.38
- K=9: 189.38
- K=10: 142.19

Observando la curva, el "codo" no es muy claro, pero podría estar alrededor de K=4 o K=6.

#### Silhouette Score

El silhouette score mide qué tan bien separados están los clusters. El score varía entre -1 y 1:
- Valores cercanos a 1: clusters bien separados y compactos
- Valores cercanos a 0: clusters superpuestos
- Valores negativos: observaciones asignadas al cluster incorrecto

![Silhouette Scores](/src/output/figures/clustering/kmeans/silhouette_scores.png)

Los silhouette scores obtenidos fueron:
- K=2: 0.3710
- K=3: 0.4206
- K=4: 0.5185
- K=5: 0.5742
- K=6: 0.6007
- K=7: 0.6336
- K=8: 0.6635
- K=9: 0.7019
- K=10: 0.7261

El silhouette score aumenta monótonamente con K, alcanzando su máximo en K=10 (0.7261).

### Candidatos seleccionados

Basado en el análisis del método del codo y silhouette score, se consideraron **K = 4, 6 y 10** como candidatos para el análisis detallado:
- **K=4**: Representa un compromiso razonable entre simplicidad y calidad de clustering
- **K=6**: Captura más estructura en los datos sin ser excesivamente complejo
- **K=10**: Obtiene el mejor silhouette score y captura estructura más fina

### Resultados para cada K

#### K = 4

![Scatter 3D K=4](/src/output/figures/clustering/kmeans/k4/scatter_3d_clusters.png)

![Biplot 3D K=4](/src/output/figures/clustering/kmeans/k4/biplot_3d_clusters.png)

**Interpretación visual**: Con K=4, se observan 4 clusters bien definidos en el espacio 3D. Los clusters parecen separarse principalmente a lo largo de PC1 (calidad del sueño/bienestar), con algunos clusters también diferenciándose en PC2 (presión arterial/edad) y PC3 (actividad física).

#### K = 6

![Scatter 3D K=6](/src/output/figures/clustering/kmeans/k6/scatter_3d_clusters.png)

![Biplot 3D K=6](/src/output/figures/clustering/kmeans/k6/biplot_3d_clusters.png)

**Interpretación visual**: Con K=6, los clusters son más granulares. Se observa que algunos clusters de K=4 se han subdividido, especialmente en las regiones de transición entre clusters. La separación entre clusters sigue siendo clara.

#### K = 10

![Scatter 3D K=10](/src/output/figures/clustering/kmeans/k10/scatter_3d_clusters.png)

![Biplot 3D K=10](/src/output/figures/clustering/kmeans/k10/biplot_3d_clusters.png)

**Interpretación visual**: Con K=10, se captura estructura más fina. Los clusters son más pequeños y específicos, lo que permite identificar subgrupos más homogéneos. Sin embargo, algunos clusters comienzan a estar más cercanos entre sí, lo que podría indicar sobre-segmentación.

---

## 5. Clustering Jerárquico

### Diferencia entre clustering aglomerativo y divisivo

El clustering jerárquico construye una jerarquía de clusters, representada típicamente mediante un dendrograma. Existen dos enfoques principales:

**Clustering aglomerativo (bottom-up)**: Comienza con cada observación como su propio cluster individual y fusiona iterativamente los clusters más cercanos hasta que todas las observaciones están en un único cluster.

**Clustering divisivo (top-down)**: Comienza con todas las observaciones en un único cluster y divide iterativamente el cluster más heterogéneo hasta que cada observación es su propio cluster.

### Criterio de linkage

En este análisis se utilizó el criterio **Ward** para el clustering aglomerativo. Ward minimiza la varianza dentro de cada cluster al fusionar, lo que tiende a producir clusters de tamaño similar y compactos.

Para el clustering divisivo se utilizó el criterio **complete linkage** (distancia máxima), que considera la distancia máxima entre pares de observaciones de diferentes clusters al decidir cuál cluster dividir.

## 5.1 Clustering Aglomerativo

### Construcción del dendrograma

El dendrograma muestra la historia de fusiones del clustering aglomerativo. Cada fusión se representa por una línea horizontal cuya altura indica la distancia entre los clusters fusionados.

![Dendrograma Aglomerativo](/src/output/figures/clustering/hierarchical/dendrogram_agglomerative_ward.png)

### Silhouette Score

![Silhouette Scores Aglomerativo](/src/output/figures/clustering/hierarchical/silhouette_scores_agglomerative_ward.png)

### Exploración de K = 4, 6 y 10

Al igual que con K-Means, se exploraron K = 4, 6 y 10 para el clustering aglomerativo.

#### K = 4

![Scatter 3D Aglomerativo K=4](/src/output/figures/clustering/hierarchical/agglomerative/k4/scatter_3d_clusters.png)

![Biplot 3D Aglomerativo K=4](/src/output/figures/clustering/hierarchical/agglomerative/k4/biplot_3d_clusters.png)

**Interpretación visual**: Los clusters aglomerativos con K=4 son muy similares a los de K-Means, lo que indica que ambos algoritmos están encontrando una estructura similar en los datos.

#### K = 6

![Scatter 3D Aglomerativo K=6](/src/output/figures/clustering/hierarchical/agglomerative/k6/scatter_3d_clusters.png)

![Biplot 3D Aglomerativo K=6](/src/output/figures/clustering/hierarchical/agglomerative/k6/biplot_3d_clusters.png)

**Interpretación visual**: Con K=6, los clusters aglomerativos continúan siendo muy similares a los de K-Means, con subdivisiones en las mismas regiones del espacio.

#### K = 10

![Scatter 3D Aglomerativo K=10](/src/output/figures/clustering/hierarchical/agglomerative/k10/scatter_3d_clusters.png)

![Biplot 3D Aglomerativo K=10](/src/output/figures/clustering/hierarchical/agglomerative/k10/biplot_3d_clusters.png)

**Interpretación visual**: Con K=10, la similitud con K-Means se mantiene, aunque se observan algunas diferencias en la forma de los clusters, especialmente en las regiones de transición.

## 5.2 Clustering Divisivo

### Construcción del dendrograma

El dendrograma divisivo muestra la historia de divisiones del clustering. Cada división se representa en el árbol, comenzando desde un único cluster raíz.

![Dendrograma Divisivo](/src/output/figures/clustering/hierarchical/dendrogram_divisive.png)

### Silhouette Score

![Silhouette Scores Divisivo](/src/output/figures/clustering/hierarchical/silhouette_scores_divisive.png)

### Exploración de K = 4, 6 y 10

#### K = 4

![Scatter 3D Divisivo K=4](/src/output/figures/clustering/hierarchical/divisive/k4/scatter_3d_clusters.png)

![Biplot 3D Divisivo K=4](/src/output/figures/clustering/hierarchical/divisive/k4/biplot_3d_clusters.png)

**Interpretación visual**: Los clusters divisivos con K=4 son diferentes a los de K-Means y aglomerativo. El enfoque top-down produce clusters con formas diferentes, especialmente en las regiones donde la densidad de puntos es más baja.

#### K = 6

![Scatter 3D Divisivo K=6](/src/output/figures/clustering/hierarchical/divisive/k6/scatter_3d_clusters.png)

![Biplot 3D Divisivo K=6](/src/output/figures/clustering/hierarchical/divisive/k6/biplot_3d_clusters.png)

**Interpretación visual**: Con K=6, las diferencias con K-Means y aglomerativo se hacen más evidentes. El clustering divisivo tiende a crear clusters más asimétricos en tamaño.

#### K = 10

![Scatter 3D Divisivo K=10](/src/output/figures/clustering/hierarchical/divisive/k10/scatter_3d_clusters.png)

![Biplot 3D Divisivo K=10](/src/output/figures/clustering/hierarchical/divisive/k10/biplot_3d_clusters.png)

**Interpretación visual**: Con K=10, el clustering divisivo produce una estructura muy diferente a los otros métodos. Algunos clusters son muy pequeños y específicos, mientras que otros son grandes y abarcan regiones extensas del espacio.

---

## 6. Clustering mediante DBSCAN

### ¿Qué es DBSCAN?

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) es un algoritmo de clustering basado en densidad. A diferencia de K-Means y el clustering jerárquico, DBSCAN:
- No requiere especificar el número de clusters previamente
- Puede identificar clusters de forma arbitraria (no necesariamente esféricos)
- Identifica automáticamente observaciones que no pertenecen a ningún cluster (ruido)

### Conceptos clave

**Densidad**: DBSCAN define un cluster como una región del espacio donde la densidad de puntos es mayor que un umbral. La densidad se mide contando cuántos puntos están dentro de un radio ε (epsilon) alrededor de cada punto.

**Parámetros**:
- **eps (epsilon)**: El radio de vecindad alrededor de cada punto
- **min_samples**: El número mínimo de puntos requeridos dentro del radio eps para que un punto sea considerado un punto central

**Diferencia con K-Means y clustering jerárquico**:
- K-Means y clustering jerárquico asumen clusters esféricos y de tamaño similar
- DBSCAN puede encontrar clusters de forma arbitraria y tamaños muy diferentes
- DBSCAN identifica ruido, mientras que K-Means y clustering jerárquico asignan todas las observaciones a algún cluster

### Selección de parámetros

Para seleccionar el valor óptimo de eps, se utilizó el método de la **k-distance plot**. Este método grafica la distancia al k-ésimo vecino más cercano para cada punto, ordenada de mayor a menor. El "codo" de esta curva indica un valor apropiado para eps.

![K-Distance Plot](/src/output/figures/clustering/dbscan/k_distance.png)

Basado en el k-distance plot con k=min_samples=5, se seleccionó:
- **min_samples = 5**
- **eps = 0.7117**

### Resultados del clustering DBSCAN

DBSCAN identificó automáticamente:
- **Número de clusters**: 14
- **Observaciones clasificadas como ruido**: 23 (6.15% de los datos)
- **Silhouette Score**: 0.8296

El hecho de que DBSCAN identificara 14 clusters (similar a K=14 en los otros métodos) sugiere que existe estructura fina en los datos que puede ser capturada por diferentes algoritmos.

### Visualizaciones

![Scatter 3D DBSCAN](/src/output/figures/clustering/dbscan/scatter_3d_clusters.png)

![Biplot 3D DBSCAN](/src/output/figures/clustering/dbscan/biplot_3d_clusters.png)

**Interpretación visual**: Los clusters de DBSCAN tienen formas más irregulares que los de K-Means y clustering jerárquico. Se observan clusters elongados y de tamaños muy diferentes. Los puntos clasificados como ruido (generalmente en color gris o negro) se encuentran en regiones de baja densidad, típicamente en las fronteras entre clusters o en regiones aisladas del espacio.

---

## 6.1 Análisis Descriptivo de Clusters DBSCAN

Si bien DBSCAN utilizó los componentes principales (PC1, PC2, PC3) para identificar los clusters, es importante analizar las características de estos clusters utilizando las **variables numéricas originales** para entender qué patrones de sueño y salud distinguen a cada grupo.

### Distribución de observaciones por cluster

DBSCAN identificó 14 clusters más 23 observaciones clasificadas como ruido (6.15% de los datos). La distribución de observaciones por cluster es:

| Cluster | N° Observaciones | % del Total |
|---------|----------------|-------------|
| Cluster_0 | 56 | 14.97% |
| Cluster_1 | 45 | 12.03% |
| Cluster_2 | 38 | 10.16% |
| Cluster_3 | 35 | 9.36% |
| Cluster_4 | 32 | 8.56% |
| Cluster_5 | 32 | 8.56% |
| Cluster_6 | 27 | 7.22% |
| Cluster_7 | 25 | 6.68% |
| Cluster_8 | 33 | 8.83% |
| Cluster_9 | 27 | 7.22% |
| Cluster_10 | 32 | 8.56% |
| Cluster_11 | 32 | 8.56% |
| Cluster_12 | 6 | 1.60% |
| Cluster_13 | 31 | 8.29% |
| Noise | 23 | 6.15% |

### Características promedio por cluster

Para interpretar qué caracteriza a cada cluster, analizamos las medias de las variables numéricas originales (en escala normalizada). Valores positivos indican valores por encima del promedio general, mientras que valores negativos indican valores por debajo del promedio.

![Heatmap de Medias por Cluster](/src/output/figures/clustering/dbscan/descriptive/heatmap_cluster_means.png)


### Radar Chart de Características

El radar chart permite visualizar de manera comparativa el perfil de cada cluster en las 9 variables numéricas:

![Radar Chart de Clusters](/src/output/figures/clustering/dbscan/descriptive/radar_chart_clusters.png)


### Distribuciones detalladas por variable

Para analizar en detalle la distribución de cada variable por cluster, se generaron boxplots y violin plots. A continuación se presentan los boxplots para las variables más relevantes:

![Boxplot Age](/src/output/figures/clustering/dbscan/descriptive/boxplot_Age.png)

![Boxplot Sleep Duration](/src/output/figures/clustering/dbscan/descriptive/boxplot_Sleep_Duration.png)

![Boxplot Quality of Sleep](/src/output/figures/clustering/dbscan/descriptive/boxplot_Quality_of_Sleep.png)

![Boxplot Stress Level](/src/output/figures/clustering/dbscan/descriptive/boxplot_Stress_Level.png)

![Boxplot Heart Rate](/src/output/figures/clustering/dbscan/descriptive/boxplot_Heart_Rate.png)

![Boxplot Daily Steps](/src/output/figures/clustering/dbscan/descriptive/boxplot_Daily_Steps.png)

### Conclusiones del análisis descriptivo

Este análisis descriptivo complementa la visualización en el espacio PCA, proporcionando una interpretación práctica de los clusters en términos de las variables originales que son más fáciles de entender desde una perspectiva de salud del sueño.

---

## 7. Comparación entre métodos

### Tabla comparativa de métricas

| Algoritmo | K | eps | min_samples | Silhouette Score | Dunn Index | Davies-Bouldin Index | N° Clusters | N° Ruido |
|-----------|---|-----|-------------|------------------|------------|---------------------|-------------|-----------|
| k-means | 4 | - | - | 0.5185 | 0.0781 | 0.8201 | 4 | 0 |
| k-means | 6 | - | - | 0.6007 | 0.1003 | 0.6346 | 6 | 0 |
| k-means | 10 | - | - | 0.7261 | 0.1107 | 0.4575 | 10 | 0 |
| k-means | 14 | - | - | 0.7877 | 0.1312 | 0.4039 | 14 | 0 |
| hierarchical_agglomerative | 4 | - | - | 0.5205 | 0.1916 | 0.7881 | 4 | 0 |
| hierarchical_agglomerative | 6 | - | - | 0.5901 | 0.1229 | 0.6078 | 6 | 0 |
| hierarchical_agglomerative | 10 | - | - | 0.7119 | 0.1321 | 0.4739 | 10 | 0 |
| hierarchical_agglomerative | 14 | - | - | 0.7682 | 0.1992 | 0.4994 | 14 | 0 |
| hierarchical_divisive | 4 | - | - | 0.4331 | 0.1275 | 1.0515 | 4 | 0 |
| hierarchical_divisive | 6 | - | - | 0.5279 | 0.1678 | 0.7607 | 6 | 0 |
| hierarchical_divisive | 10 | - | - | 0.5599 | 0.2172 | 0.6326 | 10 | 0 |
| hierarchical_divisive | 14 | - | - | 0.6723 | 0.2164 | 0.6573 | 14 | 0 |
| dbscan | 14 | 0.7117 | 5 | 0.8296 | 0.3525 | 0.2563 | 14 | 23 |

### Significado de las métricas

**Silhouette Score**: Mide qué tan bien separados están los clusters. Valores cercanos a 1 indican clusters bien separados y compactos. Valores más altos son mejores.

**Dunn Index**: Mide la relación entre la distancia mínima entre clusters y el diámetro máximo dentro de clusters. Valores más altos indican mejor separación entre clusters y clusters más compactos.

**Davies-Bouldin Index**: Mide la similitud promedio entre clusters, considerando tanto la dispersión dentro de clusters como la separación entre clusters. Valores más bajos son mejores (indica clusters más distintos y compactos).

### Análisis de fortalezas y limitaciones

#### K-Means

**Fortalezas**:
- Simple y rápido de implementar
- Escala bien a grandes conjuntos de datos
- Produce clusters de tamaño similar
- Funciona bien cuando los clusters son esféricos y de densidad similar

**Limitaciones**:
- Requiere especificar K previamente
- Asume clusters esféricos
- Sensible a outliers
- Asigna todas las observaciones a algún cluster (no identifica ruido)
- Puede converger a óptimos locales

**Resultados**: K-Means obtuvo silhouette scores que aumentan con K, alcanzando 0.7877 con K=14. Los Dunn Index son relativamente bajos, indicando que algunos clusters están relativamente cerca entre sí.

#### Clustering Jerárquico Aglomerativo

**Fortalezas**:
- No requiere especificar K previamente (el dendrograma ayuda a decidir)
- Permite explorar diferentes resoluciones de clustering
- El dendrograma proporciona información sobre la estructura de los datos
- Funciona bien con el criterio Ward para clusters compactos

**Limitaciones**:
- Más lento que K-Means para grandes conjuntos de datos
- Una vez fusionados dos clusters, no se pueden separar
- Sensible a outliers
- Asigna todas las observaciones a algún cluster

**Resultados**: El clustering aglomerativo con Ward obtuvo resultados muy similares a K-Means, con silhouette scores ligeramente menores pero Dunn Index significativamente mayores, indicando mejor separación entre clusters.

#### Clustering Jerárquico Divisivo

**Fortalezas**:
- No requiere especificar K previamente
- Puede capturar estructura que el aglomerativo podría perder
- El dendrograma proporciona información sobre la estructura

**Limitaciones**:
- Más complejo de implementar que el aglomerativo
- Computacionalmente más costoso
- Los resultados dependen fuertemente del criterio de división
- Asigna todas las observaciones a algún cluster

**Resultados**: El clustering divisivo obtuvo los peores silhouette scores entre los métodos jerárquicos, especialmente para K=4 (0.4331). Sin embargo, obtuvo Dunn Index altos, indicando buena separación entre clusters.

#### DBSCAN

**Fortalezas**:
- No requiere especificar el número de clusters
- Puede encontrar clusters de forma arbitraria (no necesariamente esféricos)
- Identifica automáticamente ruido
- Funciona bien con clusters de densidad variable
- Robusto a outliers

**Limitaciones**:
- Sensible a la selección de parámetros eps y min_samples
- Difícil cuando los clusters tienen densidades muy diferentes
- No funciona bien en espacios de alta dimensionalidad (por eso usamos PCA)
- Puede tener dificultades si la densidad de puntos varía significativamente entre clusters

**Resultados**: DBSCAN obtuvo el mejor silhouette score (0.8296) y el mejor Dunn Index (0.3525), así como el mejor Davies-Bouldin Index (0.2563). Esto indica que encontró clusters muy bien separados y compactos. La identificación de 23 puntos como ruido (6.15%) es razonable y puede representar observaciones atípicas o casos de borde.

---

## 8. Conclusiones

Respondiendo nuevamente a la pregunta inicial:

**¿Qué significa que dos personas sean similares dentro de este dataset?**

La respuesta depende del algoritmo de clustering utilizado y del criterio de similitud implícito en cada uno:

- **Para K-Means**: Dos personas son similares si están cerca en el espacio euclidiano de los componentes principales. Esto significa que tienen valores similares en las combinaciones lineales de variables que definen PC1, PC2 y PC3 (calidad del sueño/bienestar, presión arterial/edad, y actividad física).

- **Para el clustering jerárquico aglomerativo (Ward)**: La similitud se define minimizando la varianza dentro de clusters. Dos personas son similares si agruparlas no aumenta significativamente la varianza del cluster resultante. Esto produce clusters compactos y de tamaño similar.

- **Para el clustering jerárquico divisivo**: La similitud se define maximizando la distancia entre clusters al dividir. Dos personas son similares si están en el mismo subcluster después de sucesivas divisiones basadas en la distancia máxima.

- **Para DBSCAN**: Dos personas son similares si están en una región de alta densidad del espacio. Esto significa que hay al menos min_samples puntos dentro de un radio eps alrededor de cada una. DBSCAN es más flexible en cuanto a la forma de los clusters y puede identificar similitudes en regiones de forma irregular.

### Resumen de hallazgos

**PCA permitió representar las observaciones en un espacio reducido**: Los 3 primeros componentes principales capturan el 87.14% de la varianza total, permitiendo visualizar y analizar los datos en 3 dimensiones sin perder información crítica. Los componentes tienen interpretaciones claras: PC1 representa calidad del sueño/bienestar, PC2 representa presión arterial/edad, y PC3 representa actividad física.

**Cada algoritmo definió la similitud de manera distinta**:
- K-Means y el clustering jerárquico aglomerativo produjeron resultados muy similares, con clusters esféricos y de tamaño similar.
- El clustering jerárquico divisivo produjo clusters más asimétricos en tamaño y forma.
- DBSCAN produjo clusters de forma irregular y tamaños muy diferentes, además de identificar ruido.

**Qué método obtuvo los mejores indicadores**: DBSCAN obtuvo el mejor silhouette score (0.8296), el mejor Dunn Index (0.3525), y el mejor Davies-Bouldin Index (0.2563). Esto sugiere que DBSCAN encontró la estructura más natural en los datos, con clusters bien separados y compactos.

**Qué ventajas aporta DBSCAN al identificar ruido**: La capacidad de DBSCAN para identificar 23 observaciones como ruido (6.15%) es una ventaja significativa. Estas observaciones podrían representar casos atípicos, errores de medición, o individuos con patrones verdaderamente únicos que no encajan en ningún grupo principal. En aplicaciones prácticas, estos casos podrían merecer atención individualizada.

**Qué similitudes existen entre K-Means y Ward**: K-Means y el clustering jerárquico aglomerativo con Ward produjeron resultados muy similares en términos de silhouette scores y estructura de clusters. Esto no es coincidencia: ambos métodos minimizan la varianza dentro de clusters (K-Means directamente, Ward mediante el criterio de linkage). La principal diferencia es que K-Means requiere especificar K previamente, mientras que Ward permite explorar diferentes K mediante el dendrograma.

### Reflexión final

En clustering no existe una única respuesta correcta. Cada algoritmo define la similitud de manera diferente y, por lo tanto, descubre estructuras distintas en los mismos datos. La elección del método depende de:
- La naturaleza de los datos (forma y densidad de los clusters)
- Los objetivos del análisis (segmentación gruesa vs. fina)
- La importancia de identificar outliers
- La interpretabilidad deseada

En este análisis, DBSCAN obtuvo los mejores indicadores cuantitativos, pero K-Means y el clustering jerárquico aglomerativo produjeron resultados muy interpretables y similares entre sí. El clustering jerárquico divisivo, aunque obtuvo indicadores más bajos, ofrece una perspectiva diferente sobre la estructura de los datos.

La clave es entender qué criterio de similitud es más apropiado para el problema específico y seleccionar el algoritmo en consecuencia. En muchos casos, es útil aplicar múltiples métodos y comparar los resultados para obtener una comprensión más completa de la estructura de los datos.
