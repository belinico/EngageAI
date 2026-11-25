
from pathlib import Path

import joblib
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline


BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    BASE_DIR / "data/processed/segmented_customers.csv"
)

OUTPUT_PATH = (
    BASE_DIR / "data/processed/clustered_customers.csv"
)

METRICS_PATH = (
    BASE_DIR / "data/processed/clustering_metrics.csv"
)

PROFILE_PATH = (
    BASE_DIR / "data/processed/cluster_profiles.csv"
)

MODEL_DIR = BASE_DIR / "models"


FEATURES = [
    "Recency",
    "Frequency",
    "Monetary",
    "engagement_score",
    "discount_usage_rate",
    "cart_abandonment_rate"
]


def find_optimal_clusters(X):

    results = []

    for k in range(2, 9):

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        labels = model.fit_predict(X)

        score = silhouette_score(
            X,
            labels,
            sample_size=min(2000, len(X)),
            random_state=42
        )

        results.append({
            "n_clusters": k,
            "inertia": model.inertia_,
            "silhouette_score": score
        })

        print(
            f"K={k} | "
            f"Inertia={model.inertia_:.2f} | "
            f"Silhouette={score:.4f}"
        )

    metrics = pd.DataFrame(results)

    best_row = metrics.loc[
        metrics["silhouette_score"].idxmax()
    ]

    best_k = int(best_row["n_clusters"])

    return best_k, metrics


def cluster_customers(df):

    df = df.copy()

    X = df[FEATURES]

    if len(X) < 9:
        raise ValueError(
            "At least 9 customers are required."
        )

    if X.isnull().any().any():
        raise ValueError(
            "Clustering features contain missing values."
        )

    # Scale features for model selection
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("\nFinding optimal cluster count...")

    best_k, metrics = find_optimal_clusters(
        X_scaled
    )

    print(f"\nOptimal number of clusters: {best_k}")

    # Train final model
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("kmeans", KMeans(
            n_clusters=best_k,
            random_state=42,
            n_init=10
        ))
    ])

    df["cluster"] = pipeline.fit_predict(X)

    # Reduce dimensions for visualization
    scaled_features = pipeline.named_steps[
        "scaler"
    ].transform(X)

    pca = PCA(n_components=2)

    coordinates = pca.fit_transform(
        scaled_features
    )

    df["pca_1"] = coordinates[:, 0]
    df["pca_2"] = coordinates[:, 1]

    # Create customer cluster profiles
    profiles = df.groupby("cluster").agg(
        customer_count=("customer_id", "count"),
        average_recency=("Recency", "mean"),
        average_frequency=("Frequency", "mean"),
        average_monetary=("Monetary", "mean"),
        average_engagement=("engagement_score", "mean"),
        average_discount_usage=("discount_usage_rate", "mean"),
        average_cart_abandonment=("cart_abandonment_rate", "mean")
    ).round(2)

    # Save results
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(OUTPUT_PATH, index=False)
    metrics.to_csv(METRICS_PATH, index=False)
    profiles.to_csv(PROFILE_PATH)

    joblib.dump(
        pipeline,
        MODEL_DIR / "kmeans_pipeline.joblib"
    )

    joblib.dump(
        pca,
        MODEL_DIR / "pca_model.joblib"
    )

    print("\nCustomer Clustering Completed!")

    print("\nCluster Distribution:")
    print(df["cluster"].value_counts().sort_index())

    print("\nCluster Profiles:")
    print(profiles.to_string())

    print("\nModel successfully saved.")

    return df


if __name__ == "__main__":

    df = pd.read_csv(INPUT_PATH)

    cluster_customers(df)
