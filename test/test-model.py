import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
import tensorflow as tf  



def load_test_data():
    """Load x_test and y_test from the data/test folder."""
    try:
        x = pd.read_csv("data/test/x_test.csv")
        y = pd.read_csv("data/test/y_test.csv").squeeze("columns")
        return x, y
    except Exception as e:
        print("A problem occurred while importing test data:", e)
        raise


def load_model_and_type():
    """
    Load the model and detect whether it is a scikit-learn or Keras model.

    Conventions:
    - Scikit-learn model:  artifacts/Models/model.pkl   (loaded with joblib)
    - Keras model:         artifacts/Models/model.keras or model.h5
    """
    model = None
    model_type = None

    # Try to load a scikit-learn model saved with joblib
    try:
        model = joblib.load("artifacts/Models/model.pkl")
        from sklearn.base import BaseEstimator

        if isinstance(model, BaseEstimator):
            model_type = "sklearn"
            print("Detected scikit-learn model.")
            return model, model_type
    except Exception as e:
        print("Could not load scikit-learn model from artifacts/Models/model.pkl:", e)


    # Try to load a Keras model, if tensorflow is available
    keras_paths = [
        "artifacts/Models/model.keras",
        "artifacts/Models/model.h5",
        "artifacts/Models",
    ]
    for path in keras_paths:
        try:
            model = tf.keras.models.load_model(path)
            model_type = "keras"
            print(f"Detected Keras model at: {path}")
            return model, model_type
        except Exception:
                continue

    raise RuntimeError(
        "Failed to load a supported model. "
        "Expected a scikit-learn model at artifacts/Models/model.pkl "
        "or a Keras model at artifacts/Models/model(.keras|.h5)."
    )


def get_predictions(model, model_type, x_test):
    """Get class predictions from the model, adapting to sklearn or Keras."""
    try:
        if model_type == "sklearn":
            preds = model.predict(x_test)
        elif model_type == "keras":
            raw = model.predict(x_test, verbose=0)
            raw = np.asarray(raw)

            if raw.ndim == 2 and raw.shape[1] > 1:
                preds = raw.argmax(axis=1)
            else:
                preds = (raw.ravel() >= 0.5).astype(int)
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        return preds
    except Exception as e:
        print(
            "Failed to get predictions. Please check that x_test is correctly preprocessed:",
            e,
        )
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
            "Failed to compute evaluation metrics. Please check that y_test matches the model output:",
            e,
        )
        raise


def main():
    x_test, y_test = load_test_data()
    model, model_type = load_model_and_type()
    preds = get_predictions(model, model_type, x_test)
    results = evaluate_model(y_test, preds)

    with open("metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Evaluation complete.")
    print(results)


if __name__ == "__main__":
    main()