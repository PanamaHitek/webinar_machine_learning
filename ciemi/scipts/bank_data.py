import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar el archivo CSV en un DataFrame
file_path = "../datasets/bank_dataset/dataset.csv"  # Cambia esto a la ruta correcta del archivo CSV
data = pd.read_csv(file_path)

# Seleccionar solo las columnas numéricas para el cálculo de correlación
numeric_columns = data.select_dtypes(include=[float, int])

# Calcular la matriz de correlación
correlation_matrix = numeric_columns.corr()

# Configurar el estilo de la visualización
plt.figure(figsize=(10, 8))
sns.set(style="white")

# Generar un mapa de calor de la matriz de correlación
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")

# Mostrar el mapa de calor
plt.title("Matriz de Correlación")
plt.show()
