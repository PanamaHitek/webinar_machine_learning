import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.utils import all_estimators
from sklearn.metrics import r2_score
from sklearn.exceptions import NotFittedError

# Load data
df = pd.read_csv('../../datasets/house_price/dataset.csv')

# Convert categorical data to numerical data
le = preprocessing.LabelEncoder()
for column_name in df.columns:
    if df[column_name].dtype == object:
        df[column_name] = le.fit_transform(df[column_name])
    else:
        pass

# Split data into features and target
X = df.drop('price', axis=1)
y = df['price']

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Get all regressors from sklearn
regressors = all_estimators(type_filter='regressor')

# Initialize the best regressor and score
best_regressor = None
best_score = float('-inf')

# Loop over all regressors
for name, RegressorClass in regressors:
    try:
        print('Trying', name)
        reg = RegressorClass()
        reg.fit(X_train, y_train)
        predictions = reg.predict(X_test)
        score = r2_score(y_test, predictions)
        print(name, 'R^2 Score:', score)

        # If this score is the best, update best_score and best_regressor
        if score > best_score:
            best_score = score
            best_regressor = name
    except Exception as e:
        print(name, 'Error:', e)

# Print the best regressor
print('Best Regressor:', best_regressor, 'R^2 Score:', best_score)
