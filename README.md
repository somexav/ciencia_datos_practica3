# 📊 Ciencia de Datos - Práctica 3: Análisis Exploratorio

## 📝 Descripción del Proyecto

Este proyecto implementa una **librería personalizada de análisis exploratorio de datos** con funciones especializadas para preprocesamiento, visualización interactiva y análisis estadístico. La librería está diseñada para facilitar el análisis de datos médicos, específicamente datos de cardiotocografía (CTG), pero es aplicable a cualquier conjunto de datos.

## 🎯 Objetivos

- Crear una librería reutilizable para análisis exploratorio de datos
- Implementar visualizaciones interactivas con Plotly
- Aplicar técnicas de preprocesamiento y limpieza de datos
- Proporcionar análisis estadístico automatizado
- Documentar código siguiendo estándares profesionales (NumPy docstring)

## 🏗️ Estructura del Proyecto

```
ciencia_datos_practica3/
├── 📁 libreria_modulo_1/          # Librería personalizada
│   ├── __init__.py                # Configuración del módulo
│   ├── analysis.py                # Análisis estadístico
│   ├── preprocessing.py           # Preprocesamiento de datos
│   └── visualization.py           # Visualizaciones interactivas
├── 📊 CTG.csv                     # Dataset de cardiotocografía
├── 📓 practica3.ipynb            # Notebook principal
├── 📋 requirements.txt            # Dependencias del proyecto
└── 📖 README.md                   # Este archivo
```

## 🚀 Instalación

### Prerrequisitos
- Python 3.12 o superior
- pip (gestor de paquetes de Python)

### Pasos de instalación

1. **Clonar el repositorio:**
```bash
git clone https://github.com/somexav/ciencia_datos_practica3.git
cd ciencia_datos_practica3
```

2. **Instalar dependencias:**
```bash
pip install -r requirements.txt
```

3. **Verificar instalación:**
```python
import libreria_modulo_1 as lm1
print("✅ Librería instalada correctamente")
```

4. **Ejecutar pruebas unitarias:**
```bash
# Ejecutar todas las pruebas
python test_libreria_modulo_1.py

# O usar pytest si está instalado
pytest test_libreria_modulo_1.py -v
```

## 🧪 Pruebas Unitarias

El proyecto incluye **4 pruebas unitarias específicas** para las funciones clave del módulo de preprocesamiento:

### 📋 Cobertura de Pruebas

| Función | Descripción de la Prueba | Validaciones |
|---------|-------------------------|--------------|
| **`agrega_nan()`** | Agregar valores NaN aleatoriamente | ✅ Tipo de retorno, forma del DataFrame, reproducibilidad con seed |
| **`delete_missing_values()`** | Eliminar columnas con exceso de NaN | ✅ Eliminación correcta de columnas, preservación de datos válidos |
| **`impute_missing_values()`** | Imputar valores faltantes | ✅ Eliminación completa de NaN, preservación de estructura |
| **`detect_outliers_iqr()`** | Detectar outliers con método IQR | ✅ Estructura del resultado, detección correcta de outliers |

### 🎯 Tipos de Validaciones

1. **Funcionalidad Básica**: Verificación de que cada función ejecuta correctamente
2. **Tipos de Retorno**: Validación de que devuelven el tipo de datos esperado  
3. **Integridad de Datos**: Preservación de estructura y contenido válido
4. **Casos Específicos**: Validación de comportamientos particulares de cada función

### 🚀 Ejecutar las Pruebas

```bash
# Ejecutar las 4 pruebas con reporte detallado
python test_libreria_modulo_1.py

# Resultado esperado:
# 🧪 PRUEBAS UNITARIAS - MÓDULO PREPROCESSING
# ✅ 4 pruebas ejecutadas exitosamente
# 🔬 Las 4 funciones clave funcionan correctamente
```

### 📊 Estadísticas de Pruebas

- **Total de pruebas**: 4 (enfocadas en preprocesamiento)
- **Funciones cubiertas**: 4/6 funciones del módulo preprocessing
- **Tiempo de ejecución**:  30 segundos
- **Cobertura**: Funciones más críticas del flujo de trabajo

## 📚 Documentación de la Librería

### 🔧 Módulo `preprocessing`

Funciones especializadas para limpieza y preprocesamiento de datos:

| Función | Descripción | Uso Principal |
|---------|-------------|---------------|
| `agrega_nan()` | Simula valores faltantes aleatoriamente | Testing y simulación |
| `delete_missing_values()` | Elimina columnas con exceso de valores nulos | Limpieza de datos |
| `impute_missing_values()` | Imputa valores faltantes con múltiples métodos | Completitud de datos |
| `detect_outliers_iqr()` | Detecta outliers usando método IQR | Análisis de calidad |
| `detect_outliers_zscore()` | Detecta outliers usando Z-Score | Análisis estadístico |
| `remove_outliers()` | Elimina outliers del dataset | Limpieza de datos |

**Ejemplo de uso:**
```python
import libreria_modulo_1 as lb

# Imputar valores faltantes
df_clean = lb.impute_missing_values(df, method='knn')

# Detectar outliers
outliers_info = lb.detect_outliers_iqr(df, factor=1.5)
```

### 📊 Módulo `visualization`

Visualizaciones interactivas con Plotly para análisis exploratorio:

| Función | Tipo de Gráfico | Casos de Uso |
|---------|-----------------|--------------|
| `plot_interactive_histogram()` | Histograma + KDE | Distribución de variables continuas |
| `plot_interactive_boxplot()` | Boxplot interactivo | Comparación entre grupos |
| `plot_interactive_bar_horizontal()` | Barras horizontales | Variables categóricas |
| `plot_interactive_line_timeseries()` | Serie temporal | Tendencias temporales |
| `plot_interactive_dot_comparison()` | Dot plot | Comparación de dos grupos |
| `plot_interactive_density_multiclass()` | Curvas de densidad | Distribuciones por clase |
| `plot_interactive_violin_swarm()` | Gráfico de violín | Distribución y densidad |
| `plot_interactive_correlation_heatmap()` | Heatmap de correlación | Relaciones entre variables |

**Ejemplo de uso:**
```python
# Crear histograma interactivo con KDE
fig = lb.plot_interactive_histogram(df, 'variable', group_by='clase', add_kde=True)
fig.show()

# Heatmap de correlaciones
fig_corr = lb.plot_interactive_correlation_heatmap(df, method='spearman')
fig_corr.show()
```

### 📈 Módulo `analysis`

Análisis estadístico automatizado y reportes de calidad:

| Función | Propósito | Output |
|---------|-----------|--------|
| `completitud_datos()` | Evalúa porcentaje de valores nulos | Serie con % de nulos por columna |
| `check_data_completeness_JavierMartinezReyes()` | Análisis completo de datos | Diccionario con 3 DataFrames |

**Ejemplo de uso:**
```python
# Análisis rápido de completitud
nulos = lb.completitud_datos(df)

# Análisis completo
resultado = lb.check_data_completeness_JavierMartinezReyes(df)
resumen = resultado['resumen_general']
estadisticos = resultado['estadisticos_dispersion'] 
clasificacion = resultado['clasificacion_variables']
```

## 🎨 Visualizaciones Generadas

### 📊 Tipos de Gráficos Disponibles

1. **Histogramas Interactivos**
   - Distribución de variables numéricas
   - Líneas de densidad KDE superpuestas
   - Agrupación por categorías
   - Marginal boxplots

2. **Boxplots Comparativos**
   - Comparación entre grupos
   - Subgráficos por clase objetivo
   - Detección visual de outliers
   - Hover interactivo con estadísticas

3. **Gráficos de Barras Horizontales**
   - Top N categorías más frecuentes
   - Porcentajes y frecuencias absolutas
   - Colores personalizables
   - Información estadística automática

4. **Series Temporales**
   - Líneas de tendencia
   - Medias móviles
   - Simulación de fechas cuando necesario
   - Zoom y pan interactivo

5. **Dot Plots de Comparación**
   - Comparación visual entre dos grupos
   - Jitter para evitar solapamiento
   - Líneas de media automáticas
   - Hover con información detallada

6. **Curvas de Densidad Multiclase**
   - Distribuciones por clase
   - Colores diferenciados
   - Estimación de densidad de kernel (KDE)
   - Comparación visual de distribuciones

7. **Gráficos de Violín**
   - Distribución completa de datos
   - Boxplot interno integrado
   - Líneas de media y mediana
   - Control de outliers y puntos

8. **Heatmaps de Correlación**
   - Múltiples métodos de correlación
   - Anotaciones automáticas
   - Filtros de significancia
   - Escalas de color adaptativas

### 🎨 Características de las Visualizaciones

- **Interactividad completa**: Zoom, pan, hover tooltips
- **Responsividad**: Se adaptan al tamaño de pantalla
- **Personalización**: Títulos, colores, y estilos configurables
- **Estadísticas automáticas**: Información adicional integrada
- **Guardado automático**: Los gráficos se guardan automáticamente en `plots/`
- **Múltiples formatos**: HTML interactivo y PNG estático

### 💾 Guardado Automático de Gráficos

**Todas las funciones de visualización incluyen guardado automático:**

```python
# Por defecto, los gráficos se guardan automáticamente
fig = lm1.plot_interactive_histogram(df, 'columna')

# Los archivos se guardan en plots/ con timestamp
# Ejemplo: plots/histogram_columna_20251130_143022.html
#         plots/histogram_columna_20251130_143022.png

# Para desactivar el guardado automático
fig = lm1.plot_interactive_histogram(df, 'columna', save_plot=False)
```

**Formatos guardados:**
- **HTML**: Gráfico interactivo completo (recomendado para exploración)
- **PNG**: Imagen estática de alta resolución (1200x800px, 2x escala)

**Estructura de archivos:**
```
plots/
├── histogram_variable1_20251130_143022.html
├── histogram_variable1_20251130_143022.png
├── boxplot_variable2_by_categoria_20251130_143045.html
├── boxplot_variable2_by_categoria_20251130_143045.png
├── correlation_heatmap_pearson_20251130_143112.html
└── correlation_heatmap_pearson_20251130_143112.png
```


## 🧠 Recomendaciones Analíticas


1. **Preprocesamiento:**
   - Siempre evaluar completitud antes de proceder
   - Elegir método de imputación según el tipo de variable
   - Considerar el contexto del dominio al tratar outliers

2. **Visualización:**
   - Usar visualizaciones apropiadas para el tipo de dato
   - Incluir información de contexto (n, estadísticas)
   - Personalizar títulos y etiquetas descriptivas

3. **Análisis:**
   - Combinar múltiples perspectivas (univariado, bivariado)
   - Documentar decisiones de preprocesamiento
   - Validar resultados con métodos alternativos

## 💾 Dataset: Cardiotocografía (CTG)

### Descripción
El dataset contiene registros de cardiotocografía fetal, una técnica médica que monitorea la frecuencia cardíaca fetal y las contracciones uterinas durante el embarazo.

Se procesaron automáticamente 2,126 cardiotocogramas fetales (CTG) y se midieron las características diagnósticas correspondientes. Los CTG también fueron clasificados por tres obstetras expertos y se asignó una etiqueta de clasificación de consenso a cada uno de ellos. La clasificación se realizó tanto con respecto a un patrón morfológico (A, B, C, ...) como a un estado fetal (N, S, P). Por lo tanto, el dataset puede utilizarse para experimentos de 10 clases o 3 clases.

### Características del Dataset
- **Tamaño**: 2,126 registros de cardiotocogramas fetales
- **Procesamiento**: Automático con extracción de características diagnósticas
- **Clasificación experta**: Consenso de 3 obstetras especialistas
- **Clasificaciones disponibles**:
  - **Patrón morfológico**: Clases A, B, C, ... (10 clases)
  - **Estado fetal**: Normal (N), Sospechoso (S), Patológico (P) (3 clases)

### Variables Principales
- **Medidas fetales**: Frecuencia cardíaca basal, variabilidad, aceleraciones
- **Medidas uterinas**: Contracciones, intensidad, duración
- **Variables categóricas**: Clasificación de patrones, estado fetal
- **Variable objetivo**: Clase de estado fetal (Normal, Sospechoso, Patológico)

### Aplicaciones
- Predicción de complicaciones fetales
- Análisis de patrones de frecuencia cardíaca
- Evaluación de riesgo obstétrico
- Investigación en medicina perinatal
- Experimentos de clasificación multiclase (3 o 10 clases)
- Desarrollo de sistemas de apoyo al diagnóstico médico



## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver `LICENSE` para más detalles.

## 👨‍💻 Autor

**Javier Martinez Reyes**
- Programa: Diplomado en Ciencia de Datos
- Proyecto: Práctica 3 - Análisis Exploratorio
- GitHub: [@somexav](https://github.com/somexav)


