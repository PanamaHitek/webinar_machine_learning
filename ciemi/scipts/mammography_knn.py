import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

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

# Crear un clasificador KNN
knn_classifier = KNeighborsClassifier(n_neighbors=3)  # Puedes ajustar el valor de n_neighbors

# Entrenar el clasificador en los datos de entrenamiento
knn_classifier.fit(X_train_scaled, y_train)

# Realizar predicciones en los datos de prueba
y_pred = knn_classifier.predict(X_test_scaled)

# Calcular la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Precisión del modelo: {accuracy:.2f}')
