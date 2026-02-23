import os
import json
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score


TEST_DATA_PATH = "data/test/test_data.csv"
MODEL_PATH = "artifacts/Models/model.pkl"
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

    model_dict = joblib.load(MODEL_PATH)

    print("Ordinal XGBoost model loaded successfully.")
    print("-" * 50)

    return model_dict


def ordinal_predict(model_dict, X):
    m1 = model_dict["m1"]
    m2 = model_dict["m2"]
    m3 = model_dict["m3"]
    feature_order = model_dict["feature_order"]

    X = X[feature_order]

    p1 = m1.predict(X)
    p2 = m2.predict(X)
    p3 = m3.predict(X)

    final_preds = p1 + p2 + p3

    print("Prediction completed successfully.")
    print("-" * 50)

    return final_preds


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
        model_dict = load_model()
        preds = ordinal_predict(model_dict, X)
        results = evaluate_model(y, preds)

        with open(OUTPUT_METRICS_PATH, "w") as f:
            json.dump(results, f, indent=2)

        print("Metrics saved to metrics.json")

    except Exception as e:
        print("ERROR:", e)


if __name__ == "__main__":
    main()
