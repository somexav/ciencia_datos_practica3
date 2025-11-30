"""
Librería de Análisis Exploratorio de Datos
==========================================

Esta librería proporciona herramientas completas para el análisis exploratorio de datos,
incluyendo preprocesamiento, visualización y análisis estadístico.

Módulos disponibles:
- preprocessing: Funciones para limpieza y preprocesamiento de datos
- visualization: Funciones para crear visualizaciones y gráficos
- analysis: Funciones para análisis estadístico y reportes automáticos

Autor: [Tu nombre]
Versión: 1.0.0
"""

# Importar funciones principales de cada módulo
from .preprocessing import (
    agrega_nan,
    delete_missing_values,
    impute_missing_values,
    detect_outliers_iqr,
    detect_outliers_zscore,
    remove_outliers
)

from .visualization import (
    # Funciones interactivas con Plotly
    plot_interactive_histogram,
    plot_interactive_boxplot,
    plot_interactive_bar_horizontal,
    plot_interactive_line_timeseries,
    plot_interactive_dot_comparison,
    plot_interactive_density_multiclass,
    plot_interactive_violin_swarm,
    plot_interactive_correlation_heatmap
)

from .analysis import (
    completitud_datos,
    check_data_completeness_JavierMartinezReyes
)

# Información de la librería
__version__ = "1.0.0"
__author__ = "Estudiante de Data Science"
__description__ = "Librería personalizada para análisis exploratorio de datos"

# Lista de todas las funciones disponibles
__all__ = [
    # Preprocessing
    'agrega_nan',
    'delete_missing_values',
    'impute_missing_values', 
    'detect_outliers_iqr',
    'detect_outliers_zscore',
    'remove_outliers',
    
    # Visualization - Funciones interactivas con Plotly
    'plot_interactive_histogram',
    'plot_interactive_boxplot', 
    'plot_interactive_bar_horizontal',
    'plot_interactive_line_timeseries',
    'plot_interactive_dot_comparison',
    'plot_interactive_density_multiclass',
    'plot_interactive_violin_swarm',
    'plot_interactive_correlation_heatmap',
    
    # Analysis
    'completitud_datos',
    'check_data_completeness_JavierMartinezReyes'
]
