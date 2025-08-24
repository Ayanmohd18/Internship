# ================================
#      1. LIBRARY IMPORTS
# ================================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from typing import List, Tuple

# ================================
#      2. DATA HANDLING & EDA
# ================================

def load_and_summarize_data(filepath: str) -> pd.DataFrame:
    """
    Loads the dataset from a CSV and provides a basic summary.
    """
    print("--- Loading and Inspecting Data ---")
    data = pd.read_csv(filepath)
    print(f"Dataset loaded with {data.shape[0]} rows and {data.shape[1]} columns.")
    
    print("\nFirst 5 Rows:")
    print(data.head())

    print("\nData Quality Check (Missing Values):")
    if data.isnull().sum().sum() == 0:
        print("No missing values found.")
    else:
        print(data.isnull().sum())
    
    return data

def perform_exploratory_visuals(data: pd.DataFrame):
    """
    Generates and displays exploratory data analysis plots.
    """
    print("\n--- Performing Exploratory Data Analysis ---")
    
    # Pairplot to see relationships between variables
    print("Displaying pairplot to visualize feature relationships...")
    sns.pairplot(data, x_vars=['TV', 'Radio', 'Newspaper'], y_vars='Sales', height=4, aspect=1, kind='reg')
    plt.suptitle('Sales vs. Advertising Spend', y=1.02)
    plt.show()

    # Heatmap to check for correlations
    print("Displaying correlation heatmap...")
    plt.figure(figsize=(8, 6))
    sns.heatmap(data.corr(), annot=True, cmap='viridis', fmt=".2f")
    plt.title('Correlation Matrix of Features and Target')
    plt.show()

# ================================
#      3. MODELING & EVALUATION
# ================================

def build_and_evaluate_model(data: pd.DataFrame, feature_cols: List[str], target_col: str) -> Tuple[pd.Series, pd.Series]:
    """
    Splits data, trains a linear regression model, and evaluates its performance.
    """
    print("\n--- Building and Evaluating Regression Model ---")
    
    # Define features and target
    X = data[feature_cols]
    y = data[target_col]
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Data split into {len(X_train)} training samples and {len(X_test)} testing samples.")
    
    # Initialize and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("Linear Regression model trained successfully.")
    
    # Make predictions on the test set
    y_predictions = model.predict(X_test)
    
    # Evaluate the model
    r2 = r2_score(y_test, y_predictions)
    rmse = mean_squared_error(y_test, y_predictions, squared=False)
    
    print(f"\nModel Performance on Test Data:")
    print(f"R-squared (R²): {r2:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    
    return y_test, y_predictions

def visualize_model_performance(y_actual: pd.Series, y_predicted: pd.Series):
    """
    Visualizes the model's predictions against the actual values.
    """
    print("\n--- Visualizing Model Performance ---")

    # Create a comparison dataframe
    comparison_df = pd.DataFrame({'Actual Sales': y_actual, 'Predicted Sales': y_predicted})
    print("\nSample of Actual vs. Predicted Sales:")
    print(comparison_df.head())

    # Create scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(y_actual, y_predicted, alpha=0.7, edgecolors='k')
    
    # Add a line for perfect prediction
    plt.plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], 'r--', lw=2, label='Perfect Prediction')
    
    plt.xlabel("Actual Sales")
    plt.ylabel("Predicted Sales")
    plt.title("Actual vs. Predicted Sales Performance")
    plt.legend()
    plt.grid(True)
    plt.show()

# ================================
#      4. MAIN EXECUTION
# ================================

def main():
    """
    Main function to run the entire data analysis and modeling pipeline.
    """
    # Define constants
    FILEPATH = "advertising.csv"
    FEATURES = ['TV', 'Radio', 'Newspaper']
    TARGET = 'Sales'
    
    # Set plot style
    sns.set_style("whitegrid")
    
    # Run pipeline
    advertising_data = load_and_summarize_data(FILEPATH)
    perform_exploratory_visuals(advertising_data)
    y_test_actual, y_test_predicted = build_and_evaluate_model(advertising_data, FEATURES, TARGET)
    visualize_model_performance(y_test_actual, y_test_predicted)


if __name__ == "__main__":
    main()
