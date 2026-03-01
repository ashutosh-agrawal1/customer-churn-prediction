# train.py

import numpy as np
from sklearn.linear_model import LogisticRegression as SklearnLR
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from sklearn.model_selection import train_test_split

from logistic_regression_numpy import LogisticRegression as NumpyLR
from utils import load_features


def evaluate_model(name, y_true, y_pred, y_prob):
    print(f"\n===== {name} =====")

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc = roc_auc_score(y_true, y_prob)

    print(f"Accuracy : {acc * 100:.2f}%")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc:.4f}")

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print("\nConfusion Matrix:")
    print(f"""
                Predicted
                0      1
    Actual 0   {tn}    {fp}
    Actual 1   {fn}    {tp}
    """)


def main():
    np.random.seed(42)

    # -----------------------------
    # Load Data
    # -----------------------------
    X, y = load_features("Data/data_set.csv")

    # Baseline accuracy
    baseline_accuracy = max(np.mean(y), 1 - np.mean(y))
    print(f"Baseline Accuracy (Majority Class): {baseline_accuracy * 100:.2f}%")

    # -----------------------------
    # Train-Test Split (Stratified)
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # -----------------------------
    # Normalize (Train only)
    # -----------------------------
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std[std == 0] = 1  # prevent division by zero

    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std

    # =====================================================
    # 1️⃣ NumPy Logistic Regression (From Scratch)
    # =====================================================
    numpy_model = NumpyLR(iterations=2000, learning_rate=0.01)
    numpy_model.fit(X_train_norm, y_train)

    y_prob_numpy = numpy_model.predict_proba(X_test_norm)
    y_pred_numpy = numpy_model.predict(X_test_norm, threshold=0.5)

    evaluate_model(
        "NumPy Logistic Regression",
        y_test,
        y_pred_numpy,
        y_prob_numpy
    )

    # =====================================================
    # 2️⃣ Scikit-learn Logistic Regression
    # =====================================================
    sklearn_model = SklearnLR(
        max_iter=10000,
        class_weight='balanced',
        random_state=42
    )

    sklearn_model.fit(X_train_norm, y_train)

    y_prob_sklearn = sklearn_model.predict_proba(X_test_norm)[:, 1]
    y_pred_sklearn = sklearn_model.predict(X_test_norm)

    evaluate_model(
        "Scikit-learn Logistic Regression",
        y_test,
        y_pred_sklearn,
        y_prob_sklearn
    )


if __name__ == "__main__":
    main()