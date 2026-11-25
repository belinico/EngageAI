
from pathlib import Path

from src.utils.generate_dataset import (
    generate_customers,
    save_dataset
)

from src.utils.generate_products import (
    generate_products,
    save_products
)

from src.preprocessing.data_cleaner import clean_data

from src.preprocessing.feature_engineering import (
    create_features
)

from src.segmentation.rfm_analysis import calculate_rfm

from src.segmentation.kmeans_clustering import (
    cluster_customers
)

from src.recommendation.recommendation_engine import (
    generate_all_recommendations,
    save_recommendations
)


BASE_DIR = Path(__file__).resolve().parent

PROCESSED_DIR = BASE_DIR / "data/processed"


def run_pipeline():

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\n=== ENGAGE AI PIPELINE ===")

    print("\n[1/7] Generating customer dataset...")

    customers = generate_customers(5000)
    save_dataset(customers)

    print("\n[2/7] Cleaning customer data...")

    customers = clean_data(customers)

    customers.to_csv(
        PROCESSED_DIR / "cleaned_customers.csv",
        index=False
    )

    print("\n[3/7] Creating customer features...")

    customers = create_features(customers)

    customers.to_csv(
        PROCESSED_DIR / "customer_features.csv",
        index=False
    )

    print("\n[4/7] Performing RFM segmentation...")

    customers = calculate_rfm(customers)

    customers.to_csv(
        PROCESSED_DIR / "segmented_customers.csv",
        index=False
    )

    print("\n[5/7] Performing K-Means clustering...")

    customers = cluster_customers(customers)

    print("\n[6/7] Generating product catalog...")

    products = generate_products(500)
    save_products(products)

    print("\n[7/7] Generating recommendations...")

    recommendations = generate_all_recommendations(
        customers,
        products,
        top_n=5
    )

    save_recommendations(recommendations)

    print("\n=== PIPELINE COMPLETED ===")

    print(f"Total customers: {len(customers)}")
    print(f"Total products: {len(products)}")
    print(
        f"Total recommendations: "
        f"{len(recommendations)}"
    )

    return customers, products, recommendations


if __name__ == "__main__":

    run_pipeline()
