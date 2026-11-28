
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


BASE_DIR = Path(__file__).resolve().parents[2]

PROCESSED_DIR = BASE_DIR / "data/processed"
MODEL_DIR = BASE_DIR / "models"

INPUT_PATH = PROCESSED_DIR / "churn_dataset.csv"
MODEL_PATH = MODEL_DIR / "churn_model.joblib"


FEATURES = [
    "total_orders",
    "total_spent",
    "average_order_value",
    "website_visits",
    "email_open_rate",
    "discount_usage_rate",
    "days_since_last_purchase",
    "customer_lifetime_days",
    "purchase_frequency",
    "conversion_rate",
    "cart_abandonment_rate",
    "monthly_spending",
    "engagement_score"
]


def prepare_data(df):

    required_columns = FEATURES + ["churn"]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    X = df[FEATURES].replace(
        [np.inf, -np.inf],
        np.nan
    )

    y = df["churn"].astype(int)

    return X, y


def create_models():

    return {

        "Logistic Regression": Pipeline([

            ("imputer", SimpleImputer(
                strategy="median"
            )),

            ("scaler", StandardScaler()),

            ("model", LogisticRegression(
                max_iter=1000,
                random_state=42
            ))

        ]),

        "Random Forest": Pipeline([

            ("imputer", SimpleImputer(
                strategy="median"
            )),

            ("model", RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            ))

        ])
    }


def evaluate_model(model, X, y):

    predictions = model.predict(X)

    probabilities = model.predict_proba(X)[:, 1]

    return {

        "accuracy": accuracy_score(
            y, predictions
        ),

        "precision": precision_score(
            y, predictions,
            zero_division=0
        ),

        "recall": recall_score(
            y, predictions,
            zero_division=0
        ),

        "f1_score": f1_score(
            y, predictions,
            zero_division=0
        ),

        "roc_auc": roc_auc_score(
            y, probabilities
        )
    }


def train_churn_models(df):

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    X, y = prepare_data(df)

    # 80% development, 20% test
    X_dev, X_test, y_dev, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )
    )

    # 60% training, 20% validation
    X_train, X_val, y_train, y_val = (
        train_test_split(
            X_dev,
            y_dev,
            test_size=0.25,
            random_state=42,
            stratify=y_dev
        )
    )

    models = create_models()

    results = []

    best_model = None
    best_model_name = None
    best_score = -1

    print("\n=== CHURN MODEL TRAINING ===")

    print(f"Training size: {len(X_train)}")
    print(f"Validation size: {len(X_val)}")
    print(f"Test size: {len(X_test)}")

    for name, model in models.items():

        print(f"\nTraining {name}...")

        model.fit(X_train, y_train)

        metrics = evaluate_model(
            model,
            X_val,
            y_val
        )

        results.append({
            "model": name,
            "dataset": "validation",
            **metrics
        })

        for metric, value in metrics.items():

            print(f"{metric}: {value:.4f}")

        if metrics["roc_auc"] > best_score:

            best_score = metrics["roc_auc"]

            best_model = model

            best_model_name = name

    # Retrain selected model using development data
    best_model.fit(X_dev, y_dev)

    # Independent test evaluation
    test_metrics = evaluate_model(
        best_model,
        X_test,
        y_test
    )

    results.append({
        "model": best_model_name,
        "dataset": "test",
        **test_metrics
    })

    metrics_df = pd.DataFrame(results)

    metrics_df.to_csv(
        PROCESSED_DIR / "churn_model_metrics.csv",
        index=False
    )

    # Confusion matrix
    test_predictions = best_model.predict(X_test)

    matrix = confusion_matrix(
        y_test,
        test_predictions,
        labels=[0, 1]
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=[
            "Actual Active",
            "Actual Churn"
        ],
        columns=[
            "Predicted Active",
            "Predicted Churn"
        ]
    )

    matrix_df.to_csv(
        PROCESSED_DIR / "churn_confusion_matrix.csv"
    )

    # Save trained model
    artifact = {
        "model": best_model,
        "model_name": best_model_name,
        "features": FEATURES,
        "test_metrics": test_metrics
    }

    joblib.dump(
        artifact,
        MODEL_PATH
    )

    print("\n=== BEST MODEL ===")

    print(f"Model: {best_model_name}")

    print(
        f"Validation ROC-AUC: {best_score:.4f}"
    )

    print("\n=== TEST RESULTS ===")

    for metric, value in test_metrics.items():

        print(f"{metric}: {value:.4f}")

    print("\nConfusion Matrix:")

    print(matrix_df)

    print(f"\nModel saved to: {MODEL_PATH}")

    return best_model, best_model_name


def predict_churn(df, model=None):

    df = df.copy()

    if model is None:

        artifact = joblib.load(MODEL_PATH)

        model = artifact["model"]

        features = artifact["features"]

    else:

        features = FEATURES

    X = df[features].replace(
        [np.inf, -np.inf],
        np.nan
    )

    probabilities = model.predict_proba(X)[:, 1]

    df["churn_probability"] = probabilities

    df["churn_prediction"] = (
        probabilities >= 0.50
    ).astype(int)

    df["churn_risk"] = np.select(
        [
            probabilities >= 0.70,
            probabilities >= 0.30
        ],
        [
            "High Risk",
            "Medium Risk"
        ],
        default="Low Risk"
    )

    df["churn_probability"] = (
        df["churn_probability"].round(4)
    )

    return df


def save_predictions(df):

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        PROCESSED_DIR / "churn_predictions.csv"
    )

    df.to_csv(
        output_path,
        index=False
    )

    print("\n=== CHURN RISK DISTRIBUTION ===")

    print(df["churn_risk"].value_counts())

    print("\n=== HIGH RISK CUSTOMERS ===")

    columns = [
        "customer_id",
        "customer_segment",
        "churn_probability",
        "churn_risk"
    ]

    high_risk = df[
        df["churn_risk"] == "High Risk"
    ]

    print(
        high_risk[columns]
        .sort_values(
            "churn_probability",
            ascending=False
        )
        .head(10)
        .to_string(index=False)
    )

    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":

    customers = pd.read_csv(INPUT_PATH)

    model, model_name = train_churn_models(
        customers
    )

    predictions = predict_churn(
        customers,
        model
    )

    save_predictions(predictions)
