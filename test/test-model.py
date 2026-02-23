import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score


def load_test_data():
    """Load x_test and y_test from the data/test folder."""
    try:
        x = pd.read_csv("data/test/x_test.csv")
        y = pd.read_csv("data/test/y_test.csv").squeeze("columns")
        return x, y
    except Exception as e:
        print("A problem occurred while importing test data:", e)
        raise


def load_model():
    """
    Load a scikit-learn model saved with joblib.
    Supports regular models and stacking classifiers.
    """
    try:
        model = joblib.load("artifacts/Models/model.pkl")
        print("Loaded scikit-learn model successfully.")
        return model
    except Exception as e:
        raise RuntimeError(
            "Failed to load scikit-learn model from artifacts/Models/model.pkl: "
            + str(e)
        )


def get_predictions(model, x_test):
    """Get predictions from the model."""
    try:
        preds = model.predict(x_test)
        return preds
    except Exception as e:
        print("Failed to get predictions. Please check x_test and model:", e)
        raise


def evaluate_model(y_test, preds):
    """Compute evaluation metrics and return them as a dict."""
    try:
        accuracy = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="weighted")
        report = classification_report(y_test, preds)

        return {
            "accuracy": float(accuracy),
            "f1_score": float(f1),
            "classification_report": report,
        }
    except Exception as e:
        print(
            "Failed to compute evaluation metrics. Please check y_test and predictions:", e
        )
        raise


def main():
    x_test, y_test = load_test_data()
    model = load_model()
    preds = get_predictions(model, x_test)
    results = evaluate_model(y_test, preds)

    with open("metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Evaluation complete.")
    print(results)


if __name__ == "__main__":
    main()
