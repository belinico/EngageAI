
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "raw" / "customers.csv"

rng = np.random.default_rng(42)

CATEGORIES = [
    "Electronics",
    "Fashion",
    "Beauty",
    "Home & Living",
    "Sports",
    "Books",
    "Accessories"
]

CHANNELS = [
    "Email",
    "SMS",
    "Push Notification",
    "WhatsApp"
]

CITIES = [
    "Istanbul",
    "Ankara",
    "Izmir",
    "Bursa",
    "Antalya",
    "Gaziantep",
    "Adana"
]


def generate_customers(n_customers=5000):

    customers = []
    today = datetime.now().date()

    for customer_id in range(1, n_customers + 1):

        age = int(rng.integers(18, 75))

        registration_days = int(
            rng.integers(30, 1500)
        )

        registration_date = today - timedelta(
            days=registration_days
        )

        days_since_last_purchase = int(
            rng.integers(
                0, min(registration_days, 365) + 1
            )
        )

        last_purchase_date = today - timedelta(
            days=days_since_last_purchase
        )

        total_orders = int(rng.integers(1, 101))

        average_order_value = round(
            float(rng.uniform(100, 5000)), 2
        )

        total_spent = round(
            total_orders * average_order_value, 2
        )

        website_visits = int(
            rng.integers(total_orders, total_orders + 400)
        )

        cart_abandonments = int(
            rng.integers(0, min(website_visits, 50) + 1)
        )

        customer = {
            "customer_id": f"CUST{customer_id:05d}",
            "age": age,
            "city": rng.choice(CITIES),
            "registration_date": registration_date,
            "last_purchase_date": last_purchase_date,
            "total_orders": total_orders,
            "total_spent": total_spent,
            "average_order_value": average_order_value,
            "preferred_category": rng.choice(CATEGORIES),
            "website_visits": website_visits,
            "cart_abandonments": cart_abandonments,
            "email_open_rate": round(
                float(rng.beta(2, 5)), 2
            ),
            "discount_usage_rate": round(
                float(rng.beta(2, 3)), 2
            ),
            "preferred_channel": rng.choice(CHANNELS),
            "days_since_last_purchase": days_since_last_purchase
        }

        customers.append(customer)

    return pd.DataFrame(customers)


def save_dataset(df):

    DATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(DATA_PATH, index=False)

    print("Dataset successfully generated!")
    print(f"Total customers: {len(df)}")
    print(f"Total features: {len(df.columns)}")
    print(f"Saved to: {DATA_PATH}")

    print("\nFirst 5 customers:")
    print(df.head().to_string(index=False))


if __name__ == "__main__":

    df = generate_customers(5000)
    save_dataset(df)
