import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.utils import all_estimators
from sklearn.exceptions import ConvergenceWarning
import warnings

# Cargar el conjunto de datos desde el archivo CSV
data = pd.read_csv('../datasets/mammographic_mass/dataset.csv')

# Dividir el conjunto de datos en características (X) y etiquetas (y)
X = data.drop('Severity', axis=1)
y = data['Severity']

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar las características para que tengan media cero y desviación estándar unitaria
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Obtener la lista de todos los estimadores disponibles
estimators = all_estimators(type_filter='classifier')

best_accuracy = 0.0
best_estimator = None

# Iterar sobre todos los estimadores
for name, EstimatorClass in estimators:
    try:
        # Crear una instancia del estimador
        estimator = EstimatorClass()

        # Entrenar el estimador en los datos de entrenamiento
        estimator.fit(X_train_scaled, y_train)

        # Realizar predicciones en los datos de prueba
        y_pred = estimator.predict(X_test_scaled)

        # Calcular la precisión del modelo
        accuracy = accuracy_score(y_test, y_pred)

        # Actualizar el mejor estimador si se obtiene una mejor precisión
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_estimator = name

        print(f'Estimador: {name}, Precisión: {accuracy:.2f}')
    except Exception as e:
        pass

# Imprimir el mejor estimador
print(f'\nMejor estimador: {best_estimator}, Precisión: {best_accuracy:.2f}')
