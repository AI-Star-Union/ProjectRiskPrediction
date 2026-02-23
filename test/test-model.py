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
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in test data.")

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    print(f"Test data loaded: {df.shape[0]} samples, {X.shape[1]} features.")
    return X, y


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

    model_dict = joblib.load(MODEL_PATH)

    if not isinstance(model_dict, dict):
        raise ValueError("Loaded model is not a dictionary.")

    required_keys = {"m1", "m2", "m3", "feature_order"}
    if not required_keys.issubset(model_dict.keys()):
        raise ValueError("Model dictionary missing required keys.")

    print("Ordinal XGBoost model loaded successfully.")
    return model_dict


def ordinal_predict(model_dict, X):
    m1 = model_dict["m1"]
    m2 = model_dict["m2"]
    m3 = model_dict["m3"]
    feature_order = model_dict["feature_order"]

    # Ensure required features exist
    missing_features = set(feature_order) - set(X.columns)
    if missing_features:
        raise ValueError(f"Missing features in test data: {missing_features}")

    # Reorder features to match training
    X = X[feature_order]

    # Predictions
    p1 = m1.predict(X)
    p2 = m2.predict(X)
    p3 = m3.predict(X)

    final_preds = p1 + p2 + p3
    return final_preds


def evaluate_model(y, preds):
    accuracy = accuracy_score(y, preds)
    f1 = f1_score(y, preds, average="weighted")
    report = classification_report(y, preds)

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

        print("\nEvaluation complete.")
        print(json.dumps(results, indent=2))

    except Exception as e:
        print(f"\nERROR: {e}")


if __name__ == "__main__":
    main()
