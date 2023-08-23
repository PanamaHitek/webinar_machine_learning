import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Cargar el dataset
data = pd.read_csv('../datasets/mammographic_mass/dataset.csv')

# Obtener los atributos numéricos (excluyendo la columna Severity)
numeric_features = data

# Normalizar los atributos numéricos
scaler = StandardScaler()
numeric_features_scaled = scaler.fit_transform(numeric_features)

# Calcular la matriz de correlación
correlation_matrix = np.corrcoef(numeric_features_scaled, rowvar=False)

# Visualizar la matriz de correlación utilizando Seaborn
sns.set(style='white')
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm',
            xticklabels=numeric_features.columns, yticklabels=numeric_features.columns)
plt.title('Matriz de Correlación')
plt.show()
