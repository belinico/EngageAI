
from pathlib import Path

import pandas as pd
import numpy as np


BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_PATH = BASE_DIR / "data/processed/customer_features.csv"
OUTPUT_PATH = BASE_DIR / "data/processed/segmented_customers.csv"


def assign_segment(row):

    r = row["R_score"]
    f = row["F_score"]
    m = row["M_score"]

    if r >= 4 and f >= 4 and m >= 4:
        return "Champions"

    elif r >= 3 and f >= 4:
        return "Loyal Customers"

    elif r <= 2 and (f >= 3 or m >= 3):
        return "At Risk"

    elif r == 1 and f <= 2:
        return "Lost Customers"

    elif r >= 4 and f <= 2:
        return "Recent Low-Frequency Customers"

    elif r >= 3 and f >= 2:
        return "Potential Loyalists"

    elif r <= 2:
        return "Needs Attention"

    else:
        return "Regular Customers"


def calculate_rfm(df):

    df = df.copy()

    # RFM metrics
    df["Recency"] = df["days_since_last_purchase"]
    df["Frequency"] = df["total_orders"]
    df["Monetary"] = df["total_spent"]

    # Lower recency is better
    df["R_score"] = np.ceil(
        df["Recency"].rank(
            pct=True,
            ascending=False
        ) * 5
    ).clip(1, 5).astype(int)

    # Higher frequency is better
    df["F_score"] = np.ceil(
        df["Frequency"].rank(
            pct=True
        ) * 5
    ).clip(1, 5).astype(int)

    # Higher spending is better
    df["M_score"] = np.ceil(
        df["Monetary"].rank(
            pct=True
        ) * 5
    ).clip(1, 5).astype(int)

    # Total RFM score
    df["RFM_score"] = (
        df["R_score"]
        + df["F_score"]
        + df["M_score"]
    )

    # Assign customer segments
    df["customer_segment"] = df.apply(
        assign_segment,
        axis=1
    )

    return df


if __name__ == "__main__":

    df = pd.read_csv(INPUT_PATH)

    df = calculate_rfm(df)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(OUTPUT_PATH, index=False)

    print("RFM segmentation completed!")

    print("\nCustomer Segment Distribution:")
    print(df["customer_segment"].value_counts())

    print(f"\nTotal customers: {len(df)}")
    print(f"Total features: {len(df.columns)}")
    print(f"Saved to: {OUTPUT_PATH}")
