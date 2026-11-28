
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_PATH = BASE_DIR / "data/processed/clustered_customers.csv"
OUTPUT_PATH = BASE_DIR / "data/processed/churn_dataset.csv"


def create_churn_dataset(df):

    df = df.copy()

    rng = np.random.default_rng(42)

    required_columns = [
        "days_since_last_purchase",
        "purchase_frequency",
        "email_open_rate",
        "website_visits",
        "cart_abandonment_rate",
        "engagement_score"
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Normalize customer behavior
    recency_risk = (
        df["days_since_last_purchase"] / 365
    ).clip(0, 1)

    frequency_risk = np.exp(
        -df["purchase_frequency"].clip(lower=0)
    )

    email_risk = 1 - (
        df["email_open_rate"].clip(0, 1)
    )

    website_risk = np.exp(
        -df["website_visits"].clip(lower=0) / 100
    )

    abandonment_risk = (
        df["cart_abandonment_rate"].clip(0, 1)
    )

    engagement_risk = 1 - (
        df["engagement_score"] / 100
    ).clip(0, 1)

    # Simulate unpredictable customer behavior
    noise = rng.normal(0, 0.5, len(df))

    # Synthetic churn risk
    risk = (
        -3.5
        + 3.0 * recency_risk
        + 1.5 * frequency_risk
        + 1.2 * email_risk
        + 0.8 * website_risk
        + 1.5 * abandonment_risk
        + 1.2 * engagement_risk
        + noise
    )

    # Sigmoid transformation
    probability = 1 / (1 + np.exp(-risk))

    # Generate binary churn labels
    df["churn"] = rng.binomial(
        1,
        probability
    )

    return df


def save_churn_dataset(df):

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(OUTPUT_PATH, index=False)

    print("\n=== CHURN DATASET ===")

    print(f"Total customers: {len(df)}")

    print(
        f"Active customers: "
        f"{(df['churn'] == 0).sum()}"
    )

    print(
        f"Churned customers: "
        f"{(df['churn'] == 1).sum()}"
    )

    print(
        f"Churn rate: {df['churn'].mean():.2%}"
    )

    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":

    customers = pd.read_csv(INPUT_PATH)

    churn_data = create_churn_dataset(
        customers
    )

    save_churn_dataset(churn_data)
