import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config import MOVIES_PATH, CREDITS_PATH
from src.data_loader import load_and_merge_data
from src.processing import parse_json_columns, add_engineered_features, build_matrices
from src.models import calculate_weighted_rating, get_baseline_recommendations, recommendation
from src.evaluation import evaluate_model, find_best_alpha


def main():

    print("1. Wczytywanie danych...")
    df = load_and_merge_data(MOVIES_PATH, CREDITS_PATH)
    print(f"   Załadowano {len(df)} filmów.")

    print("2. Przetwarzanie danych i inżynieria cech...")
    df = parse_json_columns(df)
    df = add_engineered_features(df)

    df, C, m = calculate_weighted_rating(df)

    print("3. Budowanie macierzy cech...")
    combined_matrix, df = build_matrices(df)

    print("4. Obliczanie podobieństwa cosinusowego...")
    cosine_sim = cosine_similarity(combined_matrix)

    print("\n--- TEST BASELINE ---")
    baseline_recs = get_baseline_recommendations(df, n=5, min_votes=m)
    print(baseline_recs[['original_title', 'weighted_rating', 'vote_count']])

    print("\n--- TUNING PARAMETRÓW (Alpha) ---")
    alpha_results = find_best_alpha(df, cosine_sim)

    best_row = alpha_results.loc[alpha_results['composite_score'].idxmax()]
    best_alpha = best_row['alpha']

    print("\n📊 Wyniki Tuningu:")
    print(alpha_results[['alpha', 'quality', 'diversity', 'composite_score']])
    print(f"\n✅ Najlepsze Alpha: {best_alpha:.2f}")

    test_movie = "The Dark Knight"
    print(f"\n--- REKOMENDACJE DLA: '{test_movie}' ---")

    recs = recommendation(
        test_movie,
        cosine_sim,
        df,
        alpha=best_alpha,
        use_mmr=True
    )

    if not recs.empty:
        print(recs[['original_title', 'vote_average', 'final_score', 'genres']].head(10))
    else:
        print("Nie znaleziono filmu.")

    print("\n--- FINALNA EWALUACJA ---")
    final_stats = evaluate_model(
        df,
        cosine_sim,
        sample_size=500,
        alpha=best_alpha
    )

    print("\n📈 OSTATECZNE WYNIKI:")
    print(final_stats)


if __name__ == "__main__":
    main()