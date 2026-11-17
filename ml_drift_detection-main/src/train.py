import os
import time
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import mlflow
import mlflow.sklearn

PROCESSED_DIR = "data/processed"
MODELS_DIR = "models"
MODEL_NAME = "churn_model.pkl"


def load_data():
    X_train = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(PROCESSED_DIR, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(PROCESSED_DIR, "y_train.csv")).values.ravel()
    y_test = pd.read_csv(os.path.join(PROCESSED_DIR, "y_test.csv")).values.ravel()
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train, n_estimators=100, random_state=42):
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    predict = model.predict(X_test)
    accuracy = float(accuracy_score(y_test, predict))
    precision = float(precision_score(y_test, predict, zero_division=0))
    recall = float(recall_score(y_test, predict, zero_division=0))
    f1 = float(f1_score(y_test, predict, zero_division=0))

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
    return metrics, predict


def save_model(model):
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, MODEL_NAME)
    joblib.dump(model, model_path)
    print(f"[INFO] Model saved successfully to {model_path}")
    return model_path


def run_training_pipeline(chunk_num=1, model_version="v1.0.0", trigger="Initial Training"):
    start_time = time.time()
    X_train, X_test, y_train, y_test = load_data()

    mlflow.set_experiment("churn_prediction")
    with mlflow.start_run():
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("chunk_number", chunk_num)
        mlflow.log_param("trigger", trigger)

        trained_model = train_model(X_train, y_train)
        metrics, _ = evaluate_model(trained_model, X_test, y_test)

        mlflow.log_metric("accuracy", metrics["accuracy"])
        mlflow.log_metric("precision", metrics["precision"])
        mlflow.log_metric("recall", metrics["recall"])
        mlflow.log_metric("f1_score", metrics["f1"])

        try:
            mlflow.sklearn.log_model(
                trained_model,
                name="random_forest_model",
                skops_trusted_types=["sklearn.tree._tree.Tree"]
            )
        except Exception as e:
            print(f"[WARN] MLflow model logging fallback: {e}")

        save_model(trained_model)

    duration = time.time() - start_time
    try:
        from src.monitoring import record_model_version, record_pipeline_run
        record_model_version(
            version=model_version,
            accuracy=metrics["accuracy"],
            precision=metrics["precision"],
            recall=metrics["recall"],
            f1=metrics["f1"],
            n_samples=len(X_train) + len(X_test),
            trigger=trigger
        )
        record_pipeline_run(
            stage="Model Training",
            status="Completed",
            duration_sec=duration,
            details=metrics,
            trigger_reason=trigger
        )
    except Exception as e:
        print(f"[WARN] Failed to record monitoring state: {e}")

    return trained_model, metrics


if __name__ == "__main__":
    trained_model, metrics = run_training_pipeline()
    print(f"Training completed successfully. Metrics: {metrics}")
