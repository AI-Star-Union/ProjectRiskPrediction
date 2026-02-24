import os
import joblib
import pandas as pd
import json

from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, f1_score

TEST_DATA_PATH = "data/test/"
MODEL_PATH = "artifacts/Models/lr_model.pkl"
TARGET_COLUMN = "Risk_Level"
OUTPUT_METRICS_PATH = "metrics.json"


def load_test_data():
    """Load x_test and y_test from the data/test folder."""
    try:
        X = pd.read_csv(os.path.join(TEST_DATA_PATH, "x_test.csv"))
        y = pd.read_csv(os.path.join(TEST_DATA_PATH, "y_test.csv")).squeeze("columns")
        return X, y
    except Exception as e:
        print("Error loading test data:", e)
        raise


def load_model():
    """Load trained Pipeline model."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)

    # Ensure it is a Pipeline
    if not isinstance(model, Pipeline):
        raise TypeError("Loaded model is not a sklearn Pipeline.")

    print("Pipeline model loaded successfully.")
    print("-" * 50)

    return model


def predict(model, X):
    """Generate predictions."""
    preds = model.predict(X)

    print("Prediction completed successfully.")
    print("-" * 50)

    return preds


def evaluate_model(y_true, preds):
    """Evaluate model performance."""
    accuracy = accuracy_score(y_true, preds)
    f1 = f1_score(y_true, preds, average="weighted")
    report = classification_report(y_true, preds)

    print("Evaluation Results")
    print("-" * 50)
    print("Accuracy:", accuracy)
    print("F1 Score (Weighted):", f1)
    print("\nClassification Report:\n")
    print(report)
    print("-" * 50)

    return {
        "accuracy": float(accuracy),
        "f1_score": float(f1),
        "classification_report": report,
    }


def main():
    try:
        X, y = load_test_data()
        model = load_model()
        preds = predict(model, X)
        results = evaluate_model(y, preds)

        # Save metrics
        with open(OUTPUT_METRICS_PATH, "w") as f:
            json.dump(results, f, indent=2)

        print("Metrics saved to", OUTPUT_METRICS_PATH)

    except Exception as e:
        print("ERROR:", e)


if __name__ == "__main__":
    main()
