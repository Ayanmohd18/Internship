

# Importing the Dependencies
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn import metrics

# ===============================
# Data Collection and Processing
# ===============================

# loading the data from csv file to pandas dataframe
car_dataset = pd.read_csv('car data.csv')

# inspecting the first 5 rows of the dataframe
print(car_dataset.head())

# checking the number of rows and columns
print("Dataset shape:", car_dataset.shape)

# getting some information about the dataset
print(car_dataset.info())

# checking the number of missing values
print("Missing values:\n", car_dataset.isnull().sum())

# checking the distribution of categorical data (✅ corrected column names)
print("\nFuel_Type counts:\n", car_dataset.Fuel_Type.value_counts())
print("\nSeller_Type counts:\n", car_dataset.Seller_Type.value_counts())
print("\nTransmission counts:\n", car_dataset.Transmission.value_counts())

# ===============================
# Encoding the Categorical Data
# ===============================

# encoding "Fuel_Type" Column
car_dataset.replace({'Fuel_Type': {'Petrol': 0, 'Diesel': 1, 'CNG': 2}}, inplace=True)

# encoding "Seller_Type" Column (✅ corrected from Selling_type)
car_dataset.replace({'Seller_Type': {'Dealer': 0, 'Individual': 1}}, inplace=True)

# encoding "Transmission" Column
car_dataset.replace({'Transmission': {'Manual': 0, 'Automatic': 1}}, inplace=True)

print("\nData after encoding:\n", car_dataset.head())

# ===============================
# Splitting the data and Target
# ===============================

X = car_dataset.drop(['Car_Name', 'Selling_Price'], axis=1)
Y = car_dataset['Selling_Price']

# ===============================
# Splitting Training and Test data
# ===============================

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=2)

# ===============================
# 1. Linear Regression
# ===============================

lin_reg_model = LinearRegression()
lin_reg_model.fit(X_train, Y_train)

# ---- Model Evaluation (Training Data)
training_data_prediction = lin_reg_model.predict(X_train)
train_r2 = metrics.r2_score(Y_train, training_data_prediction)
print("\n[Linear Regression] Training R² Error:", train_r2)

plt.scatter(Y_train, training_data_prediction, color="blue")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Linear Regression - Actual vs Predicted (Training)")
plt.show()

# ---- Model Evaluation (Test Data)
test_data_prediction = lin_reg_model.predict(X_test)
test_r2 = metrics.r2_score(Y_test, test_data_prediction)
print("[Linear Regression] Test R² Error:", test_r2)

plt.scatter(Y_test, test_data_prediction, color="green")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Linear Regression - Actual vs Predicted (Test)")
plt.show()

# ===============================
# 2. Lasso Regression
# ===============================

lass_reg_model = Lasso()
lass_reg_model.fit(X_train, Y_train)

# ---- Model Evaluation (Training Data)
training_data_prediction = lass_reg_model.predict(X_train)
train_r2 = metrics.r2_score(Y_train, training_data_prediction)
print("\n[Lasso Regression] Training R² Error:", train_r2)

plt.scatter(Y_train, training_data_prediction, color="purple")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Lasso Regression - Actual vs Predicted (Training)")
plt.show()

# ---- Model Evaluation (Test Data)
test_data_prediction = lass_reg_model.predict(X_test)
test_r2 = metrics.r2_score(Y_test, test_data_prediction)
print("[Lasso Regression] Test R² Error:", test_r2)

plt.scatter(Y_test, test_data_prediction, color="orange")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Lasso Regression - Actual vs Predicted (Test)")
plt.show()
