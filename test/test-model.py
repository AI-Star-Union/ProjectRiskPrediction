import json
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score


TEST_DATA_PATH = "data/test/test_data.csv"
MODEL_PATH = "saved_models/ordinal_xgb_model.pkl"
TARGET_COLUMN = "Risk_Level"


def load_test_data():
    try:
        df = pd.read_csv(TEST_DATA_PATH)

        if TARGET_COLUMN not in df.columns:
            raise ValueError(f"Target column '{TARGET_COLUMN}' not found.")

        X = df.drop(columns=[TARGET_COLUMN])
        y = df[TARGET_COLUMN]

        print("Test data loaded successfully.")
        return X, y

    except Exception as e:
        raise RuntimeError(f"Failed to load test data: {e}")


def load_model():
    try:
        model_dict = joblib.load(MODEL_PATH)

        if not isinstance(model_dict, dict):
            raise ValueError("Loaded object is not a dictionary.")

        print("Ordinal XGBoost model loaded successfully.")
        return model_dict

    except Exception as e:
        raise RuntimeError(f"Failed to load ordinal model: {e}")


def ordinal_predict(model_dict, X):
    try:
        m1 = model_dict["m1"]
        m2 = model_dict["m2"]
        m3 = model_dict["m3"]
        feature_order = model_dict["feature_order"]

        # 🔥 Ensure feature order matches training
        X = X[feature_order]

        p1 = m1.predict(X)
        p2 = m2.predict(X)
        p3 = m3.predict(X)

        final_preds = p1 + p2 + p3

        return final_preds

    except Exception as e:
        raise RuntimeError(f"Ordinal prediction failed: {e}")


def evaluate_model(y, preds):
    try:
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
    model_dict = load_model()
    preds = ordinal_predict(model_dict, X)
    results = evaluate_model(y, preds)

    with open("metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Evaluation complete.")
    print(results)


if __name__ == "__main__":
    main()
