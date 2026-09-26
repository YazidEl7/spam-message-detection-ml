import os
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def evaluate_models():
  test_data_path = os.path.join("models", "test_data.pkl")
  if not os.path.exists(test_data_path):
    raise FileNotFoundError(
        "Test data not found. Run src/train.py first to save test set."
    )

  X_test, y_test = joblib.load(test_data_path)

  model_names = ["multinomial_nb", "logistic_regression", "linear_svc"]

  for name in model_names:
    model_path = os.path.join("models", f"{name}.pkl")
    if not os.path.exists(model_path):
      continue

    model = joblib.load(model_path)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print("=" * 40)
    print(f"Model: {name}")
    print("=" * 40)
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))
    print("\n")


if __name__ == "__main__":
  evaluate_models()
