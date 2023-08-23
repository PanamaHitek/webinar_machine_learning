import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Cargar los datos desde los archivos CSV
train_data = pd.read_csv("../datasets/mnist/mnist_train.csv")
test_data = pd.read_csv("../datasets/mnist/mnist_test.csv")

# Separar las etiquetas de los datos
X_train = train_data.iloc[:, 1:].values
y_train = train_data.iloc[:, 0].values
X_test = test_data.iloc[:, 1:].values
y_test = test_data.iloc[:, 0].values

# Normalizar los datos dividiendo por 255
X_train_normalized = X_train / 255.0
X_test_normalized = X_test / 255.0

# Crear un clasificador Random Forest con 100 árboles (puedes ajustar este valor)
random_forest_classifier = RandomForestClassifier(n_estimators=100)

# Entrenar el clasificador Random Forest con los datos de entrenamiento normalizados
print("Iniciando entrenamiento...")
for i in range(0, 100, 10):
    random_forest_classifier.set_params(n_estimators=i + 10)
    random_forest_classifier.fit(X_train_normalized, y_train)
    y_train_pred = random_forest_classifier.predict(X_train_normalized)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print(f"Árboles entrenados: {i + 10}, Precisión en entrenamiento: {train_accuracy:.4f}")

print("Entrenamiento completado.")

# Realizar predicciones en el conjunto de prueba normalizado
y_test_pred = random_forest_classifier.predict(X_test_normalized)

# Calcular la precisión en el conjunto de prueba
accuracy = accuracy_score(y_test, y_test_pred)
print("Accuracy en el conjunto de prueba:", accuracy)

