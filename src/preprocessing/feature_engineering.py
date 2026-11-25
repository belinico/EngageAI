
from pathlib import Path

import pandas as pd
import numpy as np


BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_PATH = BASE_DIR / "data/processed/cleaned_customers.csv"
OUTPUT_PATH = BASE_DIR / "data/processed/customer_features.csv"


def create_features(df):

    df = df.copy()

    df["registration_date"] = pd.to_datetime(
        df["registration_date"]
    )

    df["last_purchase_date"] = pd.to_datetime(
        df["last_purchase_date"]
    )

    reference_date = (
        df["last_purchase_date"]
        + pd.to_timedelta(
            df["days_since_last_purchase"],
            unit="D"
        )
    ).max()

    # Customer lifetime
    df["customer_lifetime_days"] = (
        reference_date - df["registration_date"]
    ).dt.days.clip(lower=1)

    lifetime_months = (
        df["customer_lifetime_days"] / 30
    ).clip(lower=1)

    # Monthly purchase frequency
    df["purchase_frequency"] = (
        df["total_orders"] / lifetime_months
    )

    # Prevent division by zero
    visits = df["website_visits"].clip(lower=1)

    # Conversion rate
    df["conversion_rate"] = (
        df["total_orders"] / visits
    ).clip(0, 1)

    # Cart abandonment rate
    df["cart_abandonment_rate"] = (
        df["cart_abandonments"] / visits
    ).clip(0, 1)

    # Monthly spending
    df["monthly_spending"] = (
        df["total_spent"] / lifetime_months
    )

    # Normalize website visits
    normalized_visits = (
        df["website_visits"]
        / max(df["website_visits"].max(), 1)
    )

    # Customer engagement score
    df["engagement_score"] = (
        normalized_visits * 0.30
        + df["email_open_rate"] * 0.30
        + df["conversion_rate"] * 0.40
    ) * 100

    df = df.replace([np.inf, -np.inf], np.nan)

    new_features = [
        "purchase_frequency",
        "conversion_rate",
        "cart_abandonment_rate",
        "monthly_spending",
        "engagement_score"
    ]

    df[new_features] = df[new_features].round(4)

    print("Feature engineering completed!")
    print(f"Dataset shape: {df.shape}")

    return df


if __name__ == "__main__":

    df = pd.read_csv(INPUT_PATH)

    df = create_features(df)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Features saved to: {OUTPUT_PATH}")
