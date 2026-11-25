
from pathlib import Path

from src.utils.generate_dataset import (
    generate_customers,
    save_dataset
)

from src.preprocessing.data_cleaner import clean_data

from src.preprocessing.feature_engineering import (
    create_features
)

from src.segmentation.rfm_analysis import calculate_rfm

from src.segmentation.kmeans_clustering import (
    cluster_customers
)


BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "data/processed"


def run_pipeline():

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\n=== ENGAGE AI PIPELINE ===")

    print("\n[1/5] Generating customer dataset...")

    df = generate_customers(5000)
    save_dataset(df)

    print("\n[2/5] Cleaning customer data...")

    df = clean_data(df)

    df.to_csv(
        PROCESSED_DIR / "cleaned_customers.csv",
        index=False
    )

    print("\n[3/5] Creating customer features...")

    df = create_features(df)

    df.to_csv(
        PROCESSED_DIR / "customer_features.csv",
        index=False
    )

    print("\n[4/5] Performing RFM segmentation...")

    df = calculate_rfm(df)

    df.to_csv(
        PROCESSED_DIR / "segmented_customers.csv",
        index=False
    )

    print("\n[5/5] Performing K-Means clustering...")

    df = cluster_customers(df)

    print("\n=== PIPELINE COMPLETED ===")

    print("\nRFM Segments:")
    print(df["customer_segment"].value_counts())

    print("\nK-Means Clusters:")
    print(df["cluster"].value_counts())

    print(f"\nTotal customers: {len(df)}")
    print(f"Total features: {len(df.columns)}")

    return df


if __name__ == "__main__":

    run_pipeline()
