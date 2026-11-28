
from pathlib import Path

from src.utils.generate_dataset import (
    generate_customers,
    save_dataset
)

from src.utils.generate_products import (
    generate_products,
    save_products
)

from src.preprocessing.data_cleaner import (
    clean_data
)

from src.preprocessing.feature_engineering import (
    create_features
)

from src.segmentation.rfm_analysis import (
    calculate_rfm
)

from src.segmentation.kmeans_clustering import (
    cluster_customers
)

from src.recommendation.recommendation_engine import (
    generate_all_recommendations,
    save_recommendations
)

from src.prediction.churn_dataset import (
    create_churn_dataset,
    save_churn_dataset
)

from src.prediction.churn_model import (
    train_churn_models,
    predict_churn,
    save_predictions
)


BASE_DIR = Path(__file__).resolve().parent

PROCESSED_DIR = BASE_DIR / "data/processed"


def run_pipeline():

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\n=== ENGAGE AI PIPELINE ===")

    # STEP 1
    print("\n[1/9] Generating customer dataset...")

    customers = generate_customers(5000)

    save_dataset(customers)

    # STEP 2
    print("\n[2/9] Cleaning customer data...")

    customers = clean_data(customers)

    customers.to_csv(
        PROCESSED_DIR / "cleaned_customers.csv",
        index=False
    )

    # STEP 3
    print("\n[3/9] Creating customer features...")

    customers = create_features(customers)

    customers.to_csv(
        PROCESSED_DIR / "customer_features.csv",
        index=False
    )

    # STEP 4
    print("\n[4/9] Performing RFM segmentation...")

    customers = calculate_rfm(customers)

    customers.to_csv(
        PROCESSED_DIR / "segmented_customers.csv",
        index=False
    )

    # STEP 5
    print("\n[5/9] Performing K-Means clustering...")

    customers = cluster_customers(customers)

    # STEP 6
    print("\n[6/9] Generating product catalog...")

    products = generate_products(500)

    save_products(products)

    # STEP 7
    print("\n[7/9] Generating recommendations...")

    recommendations = generate_all_recommendations(
        customers,
        products,
        top_n=5
    )

    save_recommendations(recommendations)

    # STEP 8
    print("\n[8/9] Preparing churn dataset...")

    churn_data = create_churn_dataset(
        customers
    )

    save_churn_dataset(churn_data)

    # STEP 9
    print("\n[9/9] Training churn prediction model...")

    churn_model, model_name = train_churn_models(
        churn_data
    )

    churn_predictions = predict_churn(
        churn_data,
        churn_model
    )

    save_predictions(churn_predictions)

    # PIPELINE SUMMARY
    print("\n=== PIPELINE COMPLETED ===")

    print(f"Total customers: {len(customers)}")

    print(f"Total products: {len(products)}")

    print(
        f"Total recommendations: "
        f"{len(recommendations)}"
    )

    print(f"Best churn model: {model_name}")

    print(
        "High-risk customers:",
        (
            churn_predictions["churn_risk"]
            == "High Risk"
        ).sum()
    )

    print(
        "Medium-risk customers:",
        (
            churn_predictions["churn_risk"]
            == "Medium Risk"
        ).sum()
    )

    print(
        "Low-risk customers:",
        (
            churn_predictions["churn_risk"]
            == "Low Risk"
        ).sum()
    )

    return (
        customers,
        products,
        recommendations,
        churn_predictions
    )


if __name__ == "__main__":

    run_pipeline()
