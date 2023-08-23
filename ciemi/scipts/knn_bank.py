import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.utils import all_estimators

# Cargar y preparar los datos
data = pd.read_csv('../datasets/bank_dataset/dataset.csv')
data.drop(columns=['CLIENTNUM'], inplace=True)

# Preprocesamiento de Datos
categorical_columns = ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']

# Codificación one-hot para variables categóricas
data_encoded = pd.get_dummies(data, columns=categorical_columns)

# División de Datos
X = data_encoded.drop(columns=['Attrition_Flag'])
y = data_encoded['Attrition_Flag']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalización de Características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Obtener todos los estimadores disponibles
estimators = all_estimators(type_filter='classifier')

best_accuracy = 0.0
best_model = None

# Iterar a través de los estimadores y evaluar su rendimiento
for name, ClassifierClass in estimators:
    try:
        classifier = ClassifierClass()
        classifier.fit(X_train_scaled, y_train)
        y_pred = classifier.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"{name} - Accuracy: {accuracy}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = name
    except Exception as e:
        pass

print(f"Best Model: {best_model} with Accuracy: {best_accuracy}")
