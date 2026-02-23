import json
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
import xgboost as xgb


TEST_DATA_PATH = "data/test/test_data.csv"
MODEL_PATH_PKL = "artifacts/Models/model.pkl"
MODEL_PATH_JSON = "artifacts/Models/model.json"
TARGET_COLUMN = "Risk_level"   # 🔴 Change if different


def load_test_data():
    """Load full test dataset and split features/target."""
    try:
        df = pd.read_csv(TEST_DATA_PATH)

        if TARGET_COLUMN not in df.columns:
            raise ValueError(
                f"Target column '{TARGET_COLUMN}' not found in test_data.csv"
            )

        X = df.drop(columns=[TARGET_COLUMN])
        y = df[TARGET_COLUMN]

        print("Test data loaded successfully.")
        return X, y

    except Exception as e:
        raise RuntimeError(f"Failed to load test data: {e}")


def load_model():
    """Load sklearn or XGBoost model."""
    try:
        model = joblib.load(MODEL_PATH_PKL)
        print("Loaded model via joblib.")
        return model
    except Exception:
        print("Joblib load failed, trying XGBoost native load...")

    try:
        model = xgb.XGBClassifier()
        model.load_model(MODEL_PATH_JSON)
        print("Loaded XGBoost model via native load.")
        return model
    except Exception as e:
        raise RuntimeError(
            f"Failed to load model from {MODEL_PATH_PKL} or {MODEL_PATH_JSON}: {e}"
        )


def evaluate_model(model, X, y):
    """Run prediction and compute metrics."""
    try:
        preds = model.predict(X)

        accuracy = accuracy_score(y, preds)
        f1 = f1_score(y, preds, average="weighted")
        report = classification_report(y, preds)

        return {
            "accuracy": float(accuracy),
            "f1_score": float(f1),
            "classification_report": report,
        }

    except Exception as e:
        raise RuntimeError(f"Model evaluation failed: {e}")


def main():
    X, y = load_test_data()
    model = load_model()
    results = evaluate_model(model, X, y)

    with open("metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Evaluation complete.")
    print(results)


if __name__ == "__main__":
    main()
