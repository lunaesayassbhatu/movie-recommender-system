"""
Movie Recommender System — Collaborative Filtering & Matrix Factorization
Author: Luna Sbahtu | Arizona State University CSE 572 Data Mining

Compares multiple recommendation algorithms on MovieLens ratings:
  - User-based Collaborative Filtering (KNN)
  - Item-based Collaborative Filtering (KNN)
  - SVD (Matrix Factorization with biases)
  - PMF (Probabilistic Matrix Factorization)
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from surprise import Dataset, Reader, accuracy
from surprise import KNNBasic, SVD
from surprise.model_selection import cross_validate, KFold, train_test_split

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
DATA_PATH    = "ratings.csv"
N_ROWS       = 100_000     # Set to None to use full dataset
TEST_SIZE    = 0.2
RANDOM_STATE = 42
N_JOBS       = -1
OUT_DIR      = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# Load Data
# ─────────────────────────────────────────────
def load_data():
    usecols = ["userId", "movieId", "rating"]
    if N_ROWS:
        df = pd.read_csv(DATA_PATH, usecols=usecols, nrows=N_ROWS)
    else:
        df = pd.read_csv(DATA_PATH, usecols=usecols)

    print(f"Loaded {len(df):,} ratings")
    print(f"Users: {df['userId'].nunique():,} | Movies: {df['movieId'].nunique():,}")
    print(f"Rating range: {df['rating'].min()} — {df['rating'].max()}")

    reader = Reader(rating_scale=(df["rating"].min(), df["rating"].max()))
    data = Dataset.load_from_df(df[["userId", "movieId", "rating"]], reader)
    return df, data


# ─────────────────────────────────────────────
# Define Models
# ─────────────────────────────────────────────
def get_models():
    return [
        ("User-based CF (cosine)",   KNNBasic(sim_options={"name": "cosine",   "user_based": True},  verbose=False)),
        ("Item-based CF (cosine)",   KNNBasic(sim_options={"name": "cosine",   "user_based": False}, verbose=False)),
        ("User-based CF (pearson)",  KNNBasic(sim_options={"name": "pearson",  "user_based": True},  verbose=False)),
        ("SVD",                      SVD(random_state=RANDOM_STATE)),
        ("PMF",                      SVD(biased=False, random_state=RANDOM_STATE)),
    ]


# ─────────────────────────────────────────────
# Cross-Validation
# ─────────────────────────────────────────────
def run_cross_validation(data, models):
    print("\n=== 5-Fold Cross-Validation ===")
    kf = KFold(n_splits=5, random_state=RANDOM_STATE, shuffle=True)
    cv_results = []

    for name, algo in models:
        t0 = time.time()
        cv = cross_validate(algo, data, measures=["RMSE", "MAE"], cv=kf, n_jobs=N_JOBS, verbose=False)
        elapsed = time.time() - t0
        row = {
            "model":      name,
            "rmse_mean":  np.mean(cv["test_rmse"]),
            "rmse_std":   np.std(cv["test_rmse"]),
            "mae_mean":   np.mean(cv["test_mae"]),
            "mae_std":    np.std(cv["test_mae"]),
            "time_s":     round(elapsed, 2)
        }
        cv_results.append(row)
        print(f"  {name:35s} RMSE={row['rmse_mean']:.4f} ± {row['rmse_std']:.4f}  MAE={row['mae_mean']:.4f}  ({elapsed:.1f}s)")

    df = pd.DataFrame(cv_results)
    df.to_csv(f"{OUT_DIR}/validation_report.csv", index=False)
    return df


# ─────────────────────────────────────────────
# Train/Test Evaluation
# ─────────────────────────────────────────────
def run_test_evaluation(data, models):
    print("\n=== Train/Test Evaluation ===")
    trainset, testset = train_test_split(data, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    test_results = []

    for name, algo in models:
        algo.fit(trainset)
        preds = algo.test(testset)
        rmse = accuracy.rmse(preds, verbose=False)
        mae  = accuracy.mae(preds,  verbose=False)
        test_results.append({"model": name, "rmse": rmse, "mae": mae})
        print(f"  {name:35s} RMSE={rmse:.4f}  MAE={mae:.4f}")

    df = pd.DataFrame(test_results)
    df.to_csv(f"{OUT_DIR}/test_report.csv", index=False)
    return df


# ─────────────────────────────────────────────
# Visualizations
# ─────────────────────────────────────────────
def plot_results(cv_df, test_df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # CV RMSE
    axes[0].barh(cv_df["model"], cv_df["rmse_mean"], xerr=cv_df["rmse_std"],
                 color="steelblue", capsize=4)
    axes[0].set_title("Cross-Validation RMSE (lower is better)")
    axes[0].set_xlabel("RMSE")

    # Test RMSE
    axes[1].barh(test_df["model"], test_df["rmse"], color="tomato")
    axes[1].set_title("Test Set RMSE (lower is better)")
    axes[1].set_xlabel("RMSE")

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/model_comparison.png", dpi=150)
    plt.show()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    df, data = load_data()
    models   = get_models()

    cv_df   = run_cross_validation(data, models)
    test_df = run_test_evaluation(data, models)
    plot_results(cv_df, test_df)

    best = test_df.loc[test_df["rmse"].idxmin(), "model"]
    print(f"\nBest model: {best}")
    print(f"Reports saved to {OUT_DIR}/")
