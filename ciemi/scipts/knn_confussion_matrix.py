import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

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

# Crear un clasificador KNN con k=3 (puedes ajustar este valor)
knn_classifier = KNeighborsClassifier(n_neighbors=3)

# Entrenar el clasificador KNN con los datos de entrenamiento normalizados
knn_classifier.fit(X_train_normalized, y_train)

# Realizar predicciones en el conjunto de prueba normalizado
y_test_pred = knn_classifier.predict(X_test_normalized)

# Calcular la precisión en el conjunto de prueba
accuracy = accuracy_score(y_test, y_test_pred)
print("Accuracy en el conjunto de prueba:", accuracy)

# Calcular la matriz de confusión
confusion_mat = confusion_matrix(y_test, y_test_pred)

# Crear un mapa de calor para visualizar la matriz de confusión
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mat, annot=True, fmt="d", cmap="Blues",
            xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Matriz de Confusión")
plt.show()
