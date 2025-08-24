
Iris Flower Species Classifier using K-Nearest Neighbors.



# ================================
#      1. IMPORT LIBRARIES
# ================================
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from typing import List, Tuple

# ================================
#      2. DATA HANDLING
# ================================

def load_iris_data(url: str, column_names: List[str]) -> pd.DataFrame:
    """
    Loads the Iris dataset from a specified URL.
    
    Args:
        url (str): The URL to the Iris dataset file.
        column_names (List[str]): A list of column names for the DataFrame.
        
    Returns:
        pd.DataFrame: The loaded dataset as a pandas DataFrame.
    """
    print("--- Loading Dataset ---")
    df = pd.read_csv(url, header=None, names=column_names)
    print(f"Dataset loaded successfully with {df.shape[0]} samples.")
    print("Sample of the first 5 rows:")
    print(df.head())
    return df

# ================================
#      3. VISUALIZATION
# ================================

def visualize_feature_distribution(df: pd.DataFrame):
    """
    Creates a pairplot to visualize relationships between features, colored by species.
    """
    print("\n--- Visualizing Feature Distributions ---")
    sns.set_style("whitegrid")
    sns.pairplot(df, hue="species", palette="husl", markers=["o", "s", "D"])
    plt.suptitle("Feature Relationships in the Iris Dataset", y=1.02)
    plt.show()

# ================================
#      4. MODEL TRAINING & PREDICTION
# ================================

def train_and_predict(features: pd.DataFrame, target: pd.Series) -> Tuple[KNeighborsClassifier, pd.Series, pd.Series]:
    """
    Splits data, trains a KNN classifier, and makes predictions.
    
    Args:
        features (pd.DataFrame): The input features for the model.
        target (pd.Series): The target labels.
        
    Returns:
        Tuple containing the trained model, actual test labels, and predicted labels.
    """
    print("\n--- Training the K-Nearest Neighbors Classifier ---")
    # Split the data into training (70%) and testing (30%) sets
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.3, random_state=42, stratify=target
    )
    print(f"Data split into {len(x_train)} training samples and {len(x_test)} test samples.")
    
    # Initialize and train the classifier
    classifier = KNeighborsClassifier(n_neighbors=3)
    classifier.fit(x_train, y_train)
    print("Model training complete.")
    
    # Make predictions on the test data
    predictions = classifier.predict(x_test)
    
    return classifier, y_test, predictions

# ================================
#      5. EVALUATION & PREDICTION
# ================================

def evaluate_model(y_true: pd.Series, y_pred: pd.Series):
    """
    Evaluates the model using accuracy and a classification report.
    """
    print("\n--- Evaluating Model Performance ---")
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    
    print(f"Accuracy on Test Data: {accuracy:.2%}")
    print("\nClassification Report:")
    print(report)

def predict_new_sample(model: KNeighborsClassifier, sample_data: dict):
    """
    Uses the trained model to predict the species for a new data sample.
    """
    print("\n--- Predicting a New Sample ---")
    new_df = pd.DataFrame(sample_data)
    prediction = model.predict(new_df)
    print(f"The provided sample data:\n{new_df}")
    print(f"Predicted Species: {prediction[0]}")

# ================================
#      6. MAIN EXECUTION
# ================================

def main():
    """
    Main function to run the Iris classification pipeline.
    """
    # Define dataset properties
    DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    COLUMN_NAMES = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
    
    # 1. Load data
    iris_df = load_iris_data(DATA_URL, COLUMN_NAMES)
    
    # 2. Visualize data
    visualize_feature_distribution(iris_df)
    
    # 3. Define features and target
    features = iris_df.drop("species", axis=1)
    target = iris_df["species"]
    
    # 4. Train the model and get predictions
    model, y_test_actual, y_test_predicted = train_and_predict(features, target)
    
    # 5. Evaluate the model's performance
    evaluate_model(y_test_actual, y_test_predicted)
    
    # 6. Predict a new, unseen sample
    new_flower_sample = {
        "sepal_length": [5.1],
        "sepal_width": [3.5],
        "petal_length": [1.4],
        "petal_width": [0.2]
    }
    predict_new_sample(model, new_flower_sample)

if __name__ == "__main__":
    main()
