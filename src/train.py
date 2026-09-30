import pandas as pd
import os 
import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

PROCESSED_DIR="data/processed"
MODELS_DIR = "models"
MODEL_NAME = "churn_model.pkl"


def load_data():
    print("loading data")
    X_train= pd.read_csv(os.path.join(PROCESSED_DIR, "X_train.csv"))
    X_test= pd.read_csv(os.path.join(PROCESSED_DIR,"X_test.csv"))

    y_train = pd.read_csv(os.path.join(PROCESSED_DIR, "y_train.csv")).values.ravel()
    y_test = pd.read_csv(os.path.join(PROCESSED_DIR, "y_test.csv")).values.ravel()
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    model= RandomForestClassifier(n_estimators= 100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model,X_test,y_test):
    print("eval model on test data")
    predict= model.predict(X_test)
    accuracy= accuracy_score(y_test, predict)

    print(f"\n--- Model Results ---")
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # The classification report gives us detailed stats like 'precision' and 'recall'
    print("\n detailed report:")
    print(classification_report(y_test, predict))

    return accuracy

def save_model(model):
    # Ensure the 'models' directory exists
    os.makedirs(MODELS_DIR, exist_ok=True)

    model_path= os.path.join(MODELS_DIR, MODEL_NAME)
    joblib.dump(model, model_path)
    print(f"[info] Model saved successfully to {model_path}")

if __name__ == "__main__":
    # Step A: Load the clean data
    X_train, X_test, y_train, y_test = load_data()
    
    mlflow.set_experiment("churn_prediction")
    with mlflow.start_run():
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("random_state", 42)

        # Step B: Train the model
        trained_model = train_model(X_train, y_train)
        
        # Step C: Evaluate how well it learned
        accuracy= evaluate_model(trained_model, X_test, y_test)
        
        mlflow.log_metric("accuracy",accuracy)
        mlflow.sklearn.log_model(trained_model,"random_forest_model")
    
        # Step D: Save the brain of the model so we don't have to retrain it every time
        save_model(trained_model)    