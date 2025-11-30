# Librería de Análisis Exploratorio de Datos

Una librería personalizada en Python para análisis exploratorio de datos con funciones especializadas para preprocesamiento, visualización y análisis estadístico.

## 📁 Estructura del Proyecto

```
libreria_modulo_1/
├── __init__.py          # Configuración del paquete
├── preprocessing.py     # Funciones de preprocesamiento
├── visualization.py     # Funciones de visualización
├── analysis.py         # Funciones de análisis estadístico
├── app.py              # Archivo de demostración
└── README.md           # Este archivo
```

## 🚀 Instalación y Uso

### Requisitos
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Importar la librería
```python
import libreria_modulo_1 as eda
import pandas as pd

# Cargar tus datos
df = pd.read_csv('tu_archivo.csv')
```

## 📊 Módulos Disponibles

### 1. Preprocessing (preprocessing.py)

#### `delete_missing_values(df, porcentage=0.2)`
Elimina filas con un porcentaje alto de valores faltantes.

```python
df_clean = eda.delete_missing_values(df, porcentage=0.2)
```

#### `impute_missing_values(df, method='auto')`
Imputa valores faltantes usando diferentes métodos.

```python
# Métodos disponibles: 'auto', 'mean', 'median', 'knn'
df_imputed = eda.impute_missing_values(df, method='auto')
```

#### `detect_outliers_iqr(df, columns=None, factor=1.5)`
Detecta outliers usando el método IQR.

```python
outliers_info = eda.detect_outliers_iqr(df, ['columna1', 'columna2'])
```

#### `remove_outliers(df, method='iqr', columns=None)`
Elimina outliers del DataFrame.

```python
df_no_outliers = eda.remove_outliers(df, method='iqr')
```

### 2. Visualization (visualization.py)

#### `plot_missing_values(df, figsize=(12, 6))`
Visualiza patrones de valores faltantes.

```python
eda.plot_missing_values(df)
```

#### `plot_distribution(df, columns=None, ncols=3)`
Gráfica distribuciones de variables numéricas.

```python
eda.plot_distribution(df, columns=['edad', 'salario'])
```

#### `plot_correlation_matrix(df, method='pearson')`
Crea matriz de correlación con heatmap.

```python
eda.plot_correlation_matrix(df, method='pearson')
```

#### `plot_boxplots(df, columns=None, ncols=3)`
Genera boxplots para detectar outliers visualmente.

```python
eda.plot_boxplots(df)
```

### 3. Analysis (analysis.py)

#### `basic_info(df)`
Muestra información básica del dataset.

```python
eda.basic_info(df)
```

#### `automated_eda_report(df)`
Genera un reporte completo automatizado.

```python
eda.automated_eda_report(df)
```

#### `generate_summary_report(df)`
Crea reporte detallado de todas las columnas.

```python
summary = eda.generate_summary_report(df)
print(summary)
```

#### `correlation_analysis(df, threshold=0.8)`
Analiza correlaciones entre variables numéricas.

```python
correlations = eda.correlation_analysis(df, threshold=0.8)
```

## 🎯 Ejemplo de Uso Completo

```python
import libreria_modulo_1 as eda
import pandas as pd

# 1. Cargar datos
df = pd.read_csv('datos.csv')

# 2. Información básica
eda.basic_info(df)

# 3. Análisis automático completo
eda.automated_eda_report(df)

# 4. Visualizar valores faltantes
eda.plot_missing_values(df)

# 5. Eliminar filas con muchos valores faltantes
df_clean = eda.delete_missing_values(df, porcentage=0.2)

# 6. Imputar valores faltantes restantes
df_imputed = eda.impute_missing_values(df_clean, method='auto')

# 7. Detectar y remover outliers
outliers_info = eda.detect_outliers_iqr(df_imputed)
df_final = eda.remove_outliers(df_imputed, method='iqr')

# 8. Visualizaciones
eda.plot_distribution(df_final)
eda.plot_correlation_matrix(df_final)
eda.plot_boxplots(df_final)
```

## 🔧 Demostración

Ejecuta `app.py` para ver una demostración completa:

```python
python app.py
```

## 📈 Características

- ✅ **Preprocesamiento automatizado**: Limpieza y tratamiento de datos
- ✅ **Múltiples métodos de imputación**: Media, mediana, moda, KNN
- ✅ **Detección de outliers**: Métodos IQR y Z-Score  
- ✅ **Visualizaciones automáticas**: Gráficos listos para usar
- ✅ **Análisis de calidad**: Detección automática de problemas
- ✅ **Reportes automatizados**: EDA completo con una función
- ✅ **Fácil de usar**: API simple e intuitiva

## 👥 Contribución

Para contribuir a esta librería:
1. Haz fork del proyecto
2. Crea una rama para tu feature
3. Haz commit de tus cambios
4. Haz push a la rama
5. Abre un Pull Request

---
**Versión**: 1.0.0  
**Autor**: Estudiante de Data Science