
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]

CUSTOMER_PATH = (
    BASE_DIR / "data/processed/clustered_customers.csv"
)

PRODUCT_PATH = (
    BASE_DIR / "data/raw/products.csv"
)

OUTPUT_PATH = (
    BASE_DIR / "data/processed/recommendations.csv"
)


WEIGHTS = {
    "category_score": 0.35,
    "budget_score": 0.25,
    "discount_score": 0.15,
    "rating_score": 0.10,
    "popularity_score": 0.10,
    "segment_score": 0.05
}


def load_data():

    customers = pd.read_csv(CUSTOMER_PATH)
    products = pd.read_csv(PRODUCT_PATH)

    return customers, products


def calculate_segment_score(customer, products):

    segment = customer["customer_segment"]

    if segment == "Champions":

        return products["rating"] / 5

    elif segment == "Loyal Customers":

        return products["popularity"] / 100

    elif segment in [
        "At Risk",
        "Lost Customers",
        "Needs Attention"
    ]:

        return (
            products["discount_rate"] / 0.40
        ).clip(0, 1)

    elif segment == "Recent Low-Frequency Customers":

        return (
            products["rating"] / 5
            + products["popularity"] / 100
        ) / 2

    else:

        return pd.Series(
            0.5,
            index=products.index
        )


def recommend_products(customer, products, top_n=5):

    # Exclude out-of-stock products
    products = products[
        products["stock"] > 0
    ].copy()

    if products.empty:
        return products

    # Category matching
    products["category_score"] = (
        products["category"]
        == customer["preferred_category"]
    ).astype(float)

    # Budget matching
    budget = max(
        float(customer["average_order_value"]),
        1
    )

    price_difference = (
        products["final_price"] - budget
    ).abs()

    products["budget_score"] = np.exp(
        -price_difference / budget
    )

    # Discount preference matching
    discount_preference = float(
        customer["discount_usage_rate"]
    )

    normalized_discount = (
        products["discount_rate"] / 0.40
    ).clip(0, 1)

    products["discount_score"] = (
        1 - discount_preference
        * (1 - normalized_discount)
    ).clip(0, 1)

    # Product quality
    products["rating_score"] = (
        products["rating"] / 5
    ).clip(0, 1)

    # Product popularity
    products["popularity_score"] = (
        products["popularity"] / 100
    ).clip(0, 1)

    # Customer segment matching
    products["segment_score"] = (
        calculate_segment_score(
            customer,
            products
        )
    )

    # Calculate weighted recommendation score
    products["recommendation_score"] = 0.0

    for feature, weight in WEIGHTS.items():

        products["recommendation_score"] += (
            products[feature] * weight
        )

    products["recommendation_score"] = (
        products["recommendation_score"] * 100
    ).round(2)

    # Rank products
    recommendations = products.sort_values(
        "recommendation_score",
        ascending=False
    ).head(top_n).copy()

    recommendations["customer_id"] = (
        customer["customer_id"]
    )

    recommendations["customer_segment"] = (
        customer["customer_segment"]
    )

    recommendations["cluster"] = customer["cluster"]

    recommendations["recommendation_rank"] = range(
        1, len(recommendations) + 1
    )

    return recommendations


def recommend_for_customer(
    customer_id,
    customers,
    products,
    top_n=5
):

    selected_customer = customers[
        customers["customer_id"] == customer_id
    ]

    if selected_customer.empty:

        raise ValueError(
            f"Customer not found: {customer_id}"
        )

    customer = selected_customer.iloc[0]

    return recommend_products(
        customer,
        products,
        top_n
    )


def generate_all_recommendations(
    customers,
    products,
    top_n=5
):

    results = []

    total = len(customers)

    for index, (_, customer) in enumerate(
        customers.iterrows(),
        start=1
    ):

        recommendations = recommend_products(
            customer,
            products,
            top_n
        )

        if not recommendations.empty:
            results.append(recommendations)

        if index % 500 == 0:

            print(
                f"Processed {index}/{total} customers"
            )

    if not results:

        raise ValueError(
            "No recommendations generated."
        )

    return pd.concat(
        results,
        ignore_index=True
    )


def save_recommendations(df):

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(OUTPUT_PATH, index=False)

    print("\nRecommendations saved!")
    print(f"Total recommendations: {len(df)}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":

    customers, products = load_data()

    print("\n=== RECOMMENDATION ENGINE ===")

    customer_id = "CUST00001"

    recommendations = recommend_for_customer(
        customer_id,
        customers,
        products
    )

    print(f"\nRecommendations for {customer_id}:")

    columns = [
        "product_name",
        "category",
        "final_price",
        "recommendation_score"
    ]

    print(
        recommendations[columns].to_string(
            index=False
        )
    )

    print("\nGenerating all recommendations...")

    all_recommendations = (
        generate_all_recommendations(
            customers,
            products
        )
    )

    save_recommendations(
        all_recommendations
    )
