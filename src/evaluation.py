import pandas as pd
import numpy as np
import random
import itertools
from collections import Counter
from src.models import recommendation

#Metrics for base model
def evaluate_baseline_model(df, top_n_movies):
    metrics = {}
    
    #Average Rating / Votes / Weighted
    metrics['avg_rating'] = top_n_movies['vote_average'].mean()
    metrics['avg_vote_count'] = top_n_movies['vote_count'].mean()
    metrics['avg_weighted_rating'] = top_n_movies['weighted_rating'].mean()
    
    #Genre Diversity
    all_genres = [g for genres in top_n_movies['genres'] for g in genres]
    metrics['genre_diversity'] = len(set(all_genres))
    metrics['most_common_genre'] = Counter(all_genres).most_common(1)[0]
    
    #Year Stats
    metrics['year_std'] = top_n_movies['release_year'].std()
    metrics['avg_year'] = top_n_movies['release_year'].mean()
    metrics['oldest_movie'] = top_n_movies['release_year'].min()
    metrics['newest_movie'] = top_n_movies['release_year'].max()
    
    #Coverage & Popularity Bias
    metrics['coverage'] = len(top_n_movies) / len(df) * 100
    metrics['popularity_bias'] = top_n_movies['vote_count'].mean() / df['vote_count'].mean()
    
    return metrics

#Calcuate some metrics for second model
def evaluate_model(data, cosine_sim, sample_size=50, top_k=10, alpha=0.3, plot_charts=False):
    metrics = {
        'quality': [], 
        'diversity': [], 
        'genre_overlap': [], 
        'popularity_bias': []
    }
    
    safe_sample = min(sample_size, len(data))
    sample_indices = random.sample(list(data.index), safe_sample)
    avg_dataset_votes = data['vote_count'].mean()

    for idx in sample_indices:
        title = data.iloc[idx]['original_title']
        
        recs = recommendation(title, cosine_sim, data, alpha=alpha, use_mmr=True)
        if recs.empty: continue
        recs = recs.head(top_k)
        
        # 1. Quality
        metrics['quality'].append(recs['vote_average'].mean())
        
        # 2. Popularity Bias
        metrics['popularity_bias'].append(recs['vote_count'].mean() / avg_dataset_votes)
        
        # 3. Genre Overlap
        input_genres = set(data.iloc[idx]['genres'])
        overlaps = []
        for _, row in recs.iterrows():
            r_genres = set(row['genres'])
            denom = len(input_genres | r_genres)
            overlaps.append(len(input_genres & r_genres) / denom if denom > 0 else 0)
        metrics['genre_overlap'].append(np.mean(overlaps))

        # 4. Diversity
        rec_idxs = recs.index.tolist()
        if len(rec_idxs) > 1:
            pairs = list(itertools.combinations(rec_idxs, 2))
            pair_sims = [cosine_sim[p[0]][p[1]] for p in pairs]
            metrics['diversity'].append(1 - np.mean(np.clip(pair_sims, 0, 1)))
        else:
            metrics['diversity'].append(0)

    return pd.DataFrame(metrics).mean()

#Function iterates over alpha values to find the best balance
def find_best_alpha(data, cosine_sim, alpha_range=np.arange(0.1, 0.9, 0.1)):
    results = []

    for alpha in alpha_range:
        metrics = evaluate_model(data, cosine_sim, sample_size=50, alpha=alpha)
        
        #Composite Score Formula
        composite = (
            0.35 * metrics['quality'] +
            0.30 * metrics['diversity'] +
            0.20 * metrics['genre_overlap'] +
            0.15 * (1 / metrics['popularity_bias'])
        )
        
        results.append({
            'alpha': alpha,
            'quality': metrics['quality'],
            'diversity': metrics['diversity'],
            'genre_overlap': metrics['genre_overlap'],
            'popularity_bias': metrics['popularity_bias'],
            'composite_score': composite
        })
        
        print(f"-> Alpha {alpha:.1f}: Composite={composite:.3f} | Div={metrics['diversity']:.3f}")

    return pd.DataFrame(results)