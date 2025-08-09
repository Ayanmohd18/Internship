# ================================
# 1. Import Libraries
# ================================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# ================================
# 2. Load Dataset
# ================================
# Example dataset: https://raw.githubusercontent.com/krishnaik06/ADV-CSV-Dataset/master/advertising.csv
df = pd.read_csv("advertising.csv")  # Place your CSV file here

print("First 5 rows of dataset:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

# ================================
# 3. Exploratory Data Analysis
# ================================
sns.pairplot(df)
plt.show()

sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.show()

# ================================
# 4. Features & Target
# ================================
X = df[['TV', 'Radio', 'Newspaper']]  # Features
y = df['Sales']  # Target

# ================================
# 5. Split Data
# ================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ================================
# 6. Train Model
# ================================
model = LinearRegression()
model.fit(X_train, y_train)

# ================================
# 7. Predictions
# ================================
y_pred = model.predict(X_test)

# ================================
# 8. Evaluation
# ================================
print("\nR² Score:", r2_score(y_test, y_pred))
print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))

# ================================
# 9. Compare Predictions
# ================================
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print("\nSample Predictions:")
print(results.head())

# ================================
# 10. Visualize Predictions
# ================================
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.show()
