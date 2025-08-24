# ===============================
#      LIBRARY IMPORTS
# ===============================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import r2_score
from typing import Any, Dict

# ===============================
#      DATA PROCESSING
# ===============================

def load_and_prepare_data(filepath: str) -> pd.DataFrame:
    """
    Loads data from a CSV file, inspects it, and encodes categorical columns.

    Args:
        filepath (str): The path to the car data CSV file.

    Returns:
        pd.DataFrame: A preprocessed DataFrame ready for modeling.
    """
    print("--- 1. Data Loading and Preparation ---")
    
    # Load the dataset
    vehicle_df = pd.read_csv(filepath)
    print(f"Dataset loaded successfully with {vehicle_df.shape[0]} rows and {vehicle_df.shape[1]} columns.")
    
    # Check for missing values
    if vehicle_df.isnull().sum().sum() == 0:
        print("Data quality check passed: No missing values found.")
    else:
        print("Warning: Missing values detected.")
        print(vehicle_df.isnull().sum())
        
    # Define encoding maps for categorical features
    encoding_maps = {
        'Fuel_Type': {'Petrol': 0, 'Diesel': 1, 'CNG': 2},
        'Seller_Type': {'Dealer': 0, 'Individual': 1},
        'Transmission': {'Manual': 0, 'Automatic': 1}
    }
    
    # Apply the encoding
    processed_df = vehicle_df.replace(encoding_maps)
    print("Categorical features encoded into numerical format.\n")
    
    return processed_df

# ===============================
#      MODELING & EVALUATION
# ===============================

def plot_predictions(y_true: pd.Series, y_pred: pd.Series, model_name: str, data_split: str):
    """
    Generates a scatter plot to visualize model predictions against actual values.

    Args:
        y_true (pd.Series): The actual target values.
        y_pred (pd.Series): The predicted values from the model.
        model_name (str): The name of the regression model.
        data_split (str): Indicates if the data is 'Training' or 'Test'.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.7, edgecolors='k', color='#3498db')
    
    # Add a 'Perfect Prediction' line for reference
    perfect_line = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(perfect_line, perfect_line, color='red', linestyle='--', linewidth=2, label='Perfect Prediction')
    
    plt.title(f'{model_name}: Actual vs. Predicted Prices ({data_split} Set)', fontsize=15)
    plt.xlabel('Actual Price (in Lakhs)', fontsize=12)
    plt.ylabel('Predicted Price (in Lakhs)', fontsize=12)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()

def train_and_evaluate_regressor(model: Any, model_name: str, X_train, Y_train, X_test, Y_test):
    """
    Trains a given regression model, evaluates its performance, and visualizes the results.

    Args:
        model (Any): An instance of a scikit-learn regressor.
        model_name (str): A user-friendly name for the model.
        X_train, Y_train: Training data and labels.
        X_test, Y_test: Testing data and labels.
    """
    print(f"--- Training and Evaluating: {model_name} ---")
    
    # Train the model
    model.fit(X_train, Y_train)
    
    # Evaluate on Training Data
    train_preds = model.predict(X_train)
    train_r2 = r2_score(Y_train, train_preds)
    print(f"Training R-squared (R²) Score: {train_r2:.4f}")
    plot_predictions(Y_train, train_preds, model_name, 'Training')

    # Evaluate on Test Data
    test_preds = model.predict(X_test)
    test_r2 = r2_score(Y_test, test_preds)
    print(f"Test R-squared (R²) Score: {test_r2:.4f}\n")
    plot_predictions(Y_test, test_preds, model_name, 'Test')

# ===============================
#        MAIN EXECUTION
# ===============================

def main():
    """
    Main function to execute the car price prediction pipeline.
    """
    # Configure plot aesthetics
    sns.set_style("whitegrid")

    # Step 1: Load and preprocess the data
    car_data = load_and_prepare_data('car data.csv')

    # Step 2: Define features (X) and target (Y)
    features = car_data.drop(['Car_Name', 'Selling_Price'], axis=1)
    target = car_data['Selling_Price']

    # Step 3: Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(features, target, test_size=0.1, random_state=42)
    print("--- 2. Data Splitting ---")
    print(f"Data split into training ({len(X_train)} samples) and test ({len(X_test)} samples) sets.\n")

    # Step 4: Define and evaluate models
    models_to_run: Dict[str, Any] = {
        "Linear Regression": LinearRegression(),
        "Lasso Regression": Lasso()
    }
    
    for name, instance in models_to_run.items():
        train_and_evaluate_regressor(instance, name, X_train, Y_train, X_test, Y_test)

if __name__ == "__main__":
    main()
