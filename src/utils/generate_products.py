
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
OUTPUT_PATH = BASE_DIR / "data/raw/products.csv"


PRODUCTS = {
    "Electronics": [
        "Wireless Headphones",
        "Smart Watch",
        "Bluetooth Speaker",
        "Keyboard",
        "Power Bank"
    ],
    "Fashion": [
        "T-Shirt",
        "Jeans",
        "Jacket",
        "Dress",
        "Sneakers"
    ],
    "Beauty": [
        "Face Serum",
        "Moisturizer",
        "Sunscreen",
        "Cleanser",
        "Perfume"
    ],
    "Home & Living": [
        "Table Lamp",
        "Coffee Machine",
        "Cushion",
        "Dinner Set",
        "Blanket"
    ],
    "Sports": [
        "Yoga Mat",
        "Running Shoes",
        "Dumbbell",
        "Sports Bag",
        "Resistance Band"
    ],
    "Books": [
        "Science Book",
        "Technology Book",
        "Novel",
        "Business Book",
        "Psychology Book"
    ],
    "Accessories": [
        "Sunglasses",
        "Wallet",
        "Handbag",
        "Necklace",
        "Watch"
    ]
}


PRICE_RANGES = {
    "Electronics": (500, 15000),
    "Fashion": (200, 5000),
    "Beauty": (100, 3500),
    "Home & Living": (200, 10000),
    "Sports": (150, 7000),
    "Books": (80, 1000),
    "Accessories": (100, 5000)
}


def generate_products(n_products=500):

    rng = np.random.default_rng(42)
    products = []

    categories = list(PRODUCTS.keys())

    for product_id in range(1, n_products + 1):

        category = str(rng.choice(categories))

        product_name = str(
            rng.choice(PRODUCTS[category])
        )

        min_price, max_price = PRICE_RANGES[category]

        price = round(
            float(rng.uniform(min_price, max_price)),
            2
        )

        discount_rate = float(
            rng.choice(
                [0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
            )
        )

        final_price = round(
            price * (1 - discount_rate),
            2
        )

        product = {
            "product_id": f"PROD{product_id:05d}",
            "product_name": f"{product_name} - {product_id}",
            "category": category,
            "price": price,
            "discount_rate": discount_rate,
            "final_price": final_price,
            "rating": round(
                float(rng.uniform(3, 5)), 1
            ),
            "stock": int(rng.integers(0, 200)),
            "popularity": int(rng.integers(1, 101))
        }

        products.append(product)

    return pd.DataFrame(products)


def save_products(df):

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(OUTPUT_PATH, index=False)

    print("Product dataset generated!")
    print(f"Total products: {len(df)}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":

    df = generate_products(500)
    save_products(df)
