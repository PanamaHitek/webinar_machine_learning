import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score

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

# Set seed for reproducibility
seed = 42

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

# Create GradientBoostingRegressor object
gbr = GradientBoostingRegressor()

# Train GradientBoostingRegressor
gbr = gbr.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = gbr.predict(X_test)

# Print the predicted and actual value for each test data point
for pred, actual in zip(y_pred, y_test):
    print('Predicted:', pred, 'Actual:', actual)

# Calculate and print R^2 score
print('R^2 Score:', r2_score(y_test, y_pred))
