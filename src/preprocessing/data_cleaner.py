
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_PATH = BASE_DIR / "data/raw/customers.csv"
OUTPUT_PATH = BASE_DIR / "data/processed/cleaned_customers.csv"


def load_data():
    return pd.read_csv(INPUT_PATH)


def clean_data(df):

    df = df.copy()

    print(f"Original dataset: {df.shape}")

    # Remove duplicate customers
    df = df.drop_duplicates(subset=["customer_id"])

    # Convert date columns
    date_columns = [
        "registration_date",
        "last_purchase_date"
    ]

    for column in date_columns:
        df[column] = pd.to_datetime(
            df[column], errors="coerce"
        )

    # Remove records with invalid essential fields
    df = df.dropna(
        subset=["customer_id"] + date_columns
    )

    # Handle missing numerical values
    numerical_columns = df.select_dtypes(
        include="number"
    ).columns

    for column in numerical_columns:
        df[column] = df[column].fillna(
            df[column].median()
        )

    # Handle missing categorical values
    categorical_columns = df.select_dtypes(
        include=["object", "string"]
    ).columns

    for column in categorical_columns:
        df[column] = df[column].fillna("Unknown")

    # Remove invalid records
    df = df[
        (df["age"] >= 18)
        & (df["age"] <= 100)
        & (df["total_spent"] >= 0)
        & (df["total_orders"] > 0)
        & (df["website_visits"] >= 0)
        & (df["cart_abandonments"] >= 0)
        & (df["days_since_last_purchase"] >= 0)
        & (df["last_purchase_date"] >= df["registration_date"])
    ]

    # Validate rate values
    rate_columns = [
        "email_open_rate",
        "discount_usage_rate"
    ]

    for column in rate_columns:
        df[column] = df[column].clip(0, 1)

    print(f"Cleaned dataset: {df.shape}")
    print(f"Remaining missing values: {df.isnull().sum().sum()}")

    return df.reset_index(drop=True)


def save_data(df):

    OUTPUT_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )

    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Cleaned dataset saved to: {OUTPUT_PATH}")


if __name__ == "__main__":

    df = load_data()
    df = clean_data(df)
    save_data(df)
