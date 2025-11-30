"""
Pruebas unitarias para el módulo de preprocesamiento
Test suite simplificado para 4 funciones clave de preprocessing
"""

import unittest
import pandas as pd
import numpy as np
import warnings
import sys
import os

# Agregar el directorio del módulo al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'libreria_modulo_1'))

# Importar módulo de preprocesamiento
from libreria_modulo_1 import preprocessing


class TestPreprocessing(unittest.TestCase):
    """Pruebas unitarias para 4 funciones clave del módulo de preprocesamiento"""
    
    def setUp(self):
        """Configurar datos de prueba antes de cada test"""
        # Dataset simple para pruebas
        self.df_test = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [10, 20, 30, 40, 50],
            'C': ['x', 'y', 'z', 'x', 'y'],
            'D': [1.1, 2.2, 3.3, 4.4, 5.5]
        })
        
        # Dataset con valores faltantes
        self.df_nan = pd.DataFrame({
            'col1': [1, 2, np.nan, 4, 5],        # 20% NaN
            'col2': [np.nan, 2, 3, np.nan, 5],   # 40% NaN
            'col3': ['a', 'b', 'c', 'd', 'e'],   # 0% NaN
            'col4': [1.0, np.nan, np.nan, np.nan, 5.0]  # 60% NaN
        })
        
        # Dataset con outliers
        self.df_outliers = pd.DataFrame({
            'normal': [1, 2, 3, 4, 5],
            'with_outliers': [1, 2, 3, 4, 100],  # 100 es outlier
            'categorical': ['A', 'B', 'C', 'D', 'E']
        })

    def test_agrega_nan_functionality(self):
        """Test 1: Función agrega_nan - Agregar valores NaN aleatoriamente"""
        print("\n🧪 Test 1: agrega_nan - Funcionalidad básica")
        
        # Test con parámetros válidos
        result = preprocessing.agrega_nan(self.df_test, min_frac=0.1, max_frac=0.3, seed=42)
        
        # Verificar que el resultado es un DataFrame
        self.assertIsInstance(result, pd.DataFrame)
        print("   ✅ Retorna un DataFrame")
        
        # Verificar que tiene la misma forma que el original
        self.assertEqual(result.shape, self.df_test.shape)
        print(f"   ✅ Mantiene la forma original: {result.shape}")
        
        # Verificar que se agregaron algunos NaN
        original_nan_count = self.df_test.isnull().sum().sum()
        result_nan_count = result.isnull().sum().sum()
        self.assertGreater(result_nan_count, original_nan_count)
        print(f"   ✅ Agregó NaN: {original_nan_count} → {result_nan_count}")
        
        # Test con seed para reproducibilidad
        result1 = preprocessing.agrega_nan(self.df_test, min_frac=0.2, max_frac=0.2, seed=42)
        result2 = preprocessing.agrega_nan(self.df_test, min_frac=0.2, max_frac=0.2, seed=42)
        pd.testing.assert_frame_equal(result1, result2)
        print("   ✅ Reproducible con seed")

    def test_delete_missing_values_functionality(self):
        """Test 2: Función delete_missing_values - Eliminar columnas con muchos NaN"""
        print("\n🧪 Test 2: delete_missing_values - Eliminar columnas")
        
        # Test eliminando columnas con más del 50% de NaN
        # Suprimir el output de la función
        import io
        from contextlib import redirect_stdout
        
        with redirect_stdout(io.StringIO()):
            result = preprocessing.delete_missing_values(self.df_nan, porcentage=0.5)
        
        # col4 tiene 60% NaN, debería ser eliminada
        self.assertNotIn('col4', result.columns)
        print("   ✅ Eliminó col4 (60% NaN)")
        
        # col1, col2, col3 deberían mantenerse
        self.assertIn('col1', result.columns)  # col1 tiene 20% NaN
        self.assertIn('col2', result.columns)  # col2 tiene 40% NaN  
        self.assertIn('col3', result.columns)  # col3 no tiene NaN
        print("   ✅ Mantuvo columnas con <50% NaN")
        
        # Verificar forma final
        expected_cols = 3  # col1, col2, col3
        self.assertEqual(len(result.columns), expected_cols)
        print(f"   ✅ Forma final: {result.shape}")

    def test_impute_missing_values_functionality(self):
        """Test 3: Función impute_missing_values - Imputar valores faltantes"""
        print("\n🧪 Test 3: impute_missing_values - Imputación de valores")
        
        # Suprimir el output de la función
        import io
        from contextlib import redirect_stdout
        
        with redirect_stdout(io.StringIO()):
            result = preprocessing.impute_missing_values(self.df_nan.copy(), method='mean')
        
        # Verificar que es un DataFrame
        self.assertIsInstance(result, pd.DataFrame)
        print("   ✅ Retorna un DataFrame")
        
        # Verificar que no quedan valores faltantes en variables numéricas
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        nan_count_numeric = result[numeric_cols].isnull().sum().sum()
        self.assertEqual(nan_count_numeric, 0)
        print(f"   ✅ Imputó todos los NaN en columnas numéricas")
        
        # Verificar que mantiene la forma original
        self.assertEqual(result.shape, self.df_nan.shape)
        print(f"   ✅ Mantiene forma original: {result.shape}")
        
        # Verificar que los valores imputados son razonables (no NaN)
        self.assertFalse(result['col1'].isnull().any())
        self.assertFalse(result['col2'].isnull().any())
        self.assertFalse(result['col4'].isnull().any())
        print("   ✅ Valores imputados son válidos")

    def test_detect_outliers_iqr_functionality(self):
        """Test 4: Función detect_outliers_iqr - Detectar outliers usando IQR"""
        print("\n🧪 Test 4: detect_outliers_iqr - Detección de outliers")
        
        result = preprocessing.detect_outliers_iqr(self.df_outliers, columns=['with_outliers', 'normal'])
        
        # Verificar que el resultado es un DataFrame
        self.assertIsInstance(result, pd.DataFrame)
        print("   ✅ Retorna un DataFrame")
        
        # Verificar que tiene las columnas esperadas
        expected_columns = ['columna', 'Q1', 'Q3', 'IQR', 'limite_inferior', 
                          'limite_superior', 'num_outliers', 'porcentaje_outliers']
        for col in expected_columns:
            self.assertIn(col, result.columns)
        print("   ✅ Contiene todas las columnas esperadas")
        
        # Verificar que detecta el outlier en 'with_outliers'
        outlier_info = result[result['columna'] == 'with_outliers'].iloc[0]
        self.assertGreater(outlier_info['num_outliers'], 0)
        print(f"   ✅ Detectó {outlier_info['num_outliers']} outlier(s) en 'with_outliers'")
        
        # Verificar que no detecta outliers en 'normal'
        normal_info = result[result['columna'] == 'normal'].iloc[0]
        self.assertEqual(normal_info['num_outliers'], 0)
        print(f"   ✅ No detectó outliers en 'normal' (correcto)")
        
        # Verificar que analiza el número correcto de columnas
        self.assertEqual(len(result), 2)  # Debería analizar 2 columnas
        print("   ✅ Analizó el número correcto de columnas")


def run_simplified_tests():
    """Ejecutar las 4 pruebas simplificadas con reporte detallado"""
    print("🧪 PRUEBAS UNITARIAS - MÓDULO PREPROCESSING")
    print("=" * 60)
    print("📋 Ejecutando 4 pruebas clave para funciones de preprocesamiento")
    print("=" * 60)
    
    # Configurar warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    
    # Crear y ejecutar suite de pruebas
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestPreprocessing)
    
    # Ejecutar con más verbosidad
    runner = unittest.TextTestRunner(verbosity=0, stream=open(os.devnull, 'w'))
    result = runner.run(suite)
    
    # Reporte manual más claro
    print(f"\n{'='*60}")
    print("📊 RESUMEN DE RESULTADOS")
    print(f"{'='*60}")
    print(f"✅ Pruebas ejecutadas: {result.testsRun}")
    print(f"✅ Pruebas exitosas: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"❌ Fallas: {len(result.failures)}")
    print(f"💥 Errores: {len(result.errors)}")
    
    if result.failures:
        print(f"\n🔴 FALLAS ENCONTRADAS:")
        for i, (test, traceback) in enumerate(result.failures, 1):
            print(f"{i}. {test}")
            print(f"   💡 {traceback.split('AssertionError: ')[-1].split('\\n')[0]}")
    
    if result.errors:
        print(f"\n🔴 ERRORES ENCONTRADOS:")
        for i, (test, traceback) in enumerate(result.errors, 1):
            print(f"{i}. {test}")
            print(f"   💥 {traceback.split('\\n')[-2]}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    
    if success:
        print(f"\n🎉 ¡TODAS LAS PRUEBAS PASARON EXITOSAMENTE!")
        print("🔬 Las 4 funciones clave de preprocesamiento funcionan correctamente")
    else:
        print(f"\n⚠️  Algunas pruebas fallaron. Revisar implementación.")
    
    print(f"{'='*60}")
    return success


if __name__ == '__main__':
    print(f"📅 Fecha de ejecución: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    success = run_simplified_tests()
    sys.exit(0 if success else 1)