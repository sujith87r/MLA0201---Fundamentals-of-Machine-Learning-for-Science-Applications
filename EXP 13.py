import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load and preprocess the data
# Assume you have a CSV file 'car_data.csv' with relevant car features and prices
data = pd.read_csv('car_data.csv')

# Handle missing values if any
data = data.dropna()

# Convert categorical data into numerical using one-hot encoding
data = pd.get_dummies(data, columns=['make', 'model', 'year', 'fuel_type', 'transmission'])

# Step 2: Split the data into features (X) and target variable (y)
X = data.drop('price', axis=1)
y = data['price']

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Create and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Make predictions on the test set
y_pred = model.predict(X_test)

# Step 6: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R-squared:", r2)
