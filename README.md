# Movie Recommender System

A collaborative filtering and matrix factorization system built on the MovieLens dataset, comparing multiple recommendation algorithms using 5-fold cross-validation.

## Models Compared

| Model | Type | Description |
|-------|------|-------------|
| User-based CF | Collaborative Filtering | Recommends based on similar users |
| Item-based CF | Collaborative Filtering | Recommends based on similar items |
| SVD | Matrix Factorization | Singular Value Decomposition with biases |
| PMF | Probabilistic MF | SVD without biases — pure latent factors |

## Similarity Metrics (KNN)
- Cosine similarity
- Pearson correlation
- MSD (Mean Squared Difference)

## Evaluation

- **5-fold cross-validation** on full dataset
- **Train/test split** (80/20) for final evaluation
- Metrics: RMSE and MAE

## Dataset

[MovieLens](https://grouplens.org/datasets/movielens/) ratings dataset — userId, movieId, rating, timestamp.

> Download `ratings.csv` from MovieLens and place it in the project root before running.

## Usage

```bash
pip install -r requirements.txt
jupyter notebook recommender_system.ipynb
```

## Tech Stack

- Python
- scikit-surprise
- pandas
- NumPy
