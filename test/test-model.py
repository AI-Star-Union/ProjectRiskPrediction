import os
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.linear_model import LogisticRegression

TEST_DATA_PATH = "data/test/test_data.csv"
MODEL_PATH = "artifacts/Models/logistic_model.pkl"  # Change to your LR model path
TARGET_COLUMN = "Risk_Level"
OUTPUT_METRICS_PATH = "metrics.json"


def load_test_data():
    if not os.path.exists(TEST_DATA_PATH):
        raise FileNotFoundError(f"Test data not found at {TEST_DATA_PATH}")

    df = pd.read_csv(TEST_DATA_PATH)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found.")

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    print("Test data loaded successfully.")
    print("Samples:", len(df))
    print("Features:", X.shape[1])
    print("-" * 50)

    return X, y


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)

    if not isinstance(model, LogisticRegression):
        raise TypeError("Loaded model is not a LogisticRegression instance.")

    print("Logistic Regression model loaded successfully.")
    print("-" * 50)

    return model


def predict(model, X):
    preds = model.predict(X)

    print("Prediction completed successfully.")
    print("-" * 50)

    return preds


def evaluate_model(y, preds):
    accuracy = accuracy_score(y, preds)
    f1 = f1_score(y, preds, average="weighted")
    report = classification_report(y, preds)

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

        # Save metrics if needed
        with open(OUTPUT_METRICS_PATH, "w") as f:
            import json
            json.dump(results, f, indent=2)

        print("Metrics saved to", OUTPUT_METRICS_PATH)

    except Exception as e:
        print("ERROR:", e)


if __name__ == "__main__":
    main()
