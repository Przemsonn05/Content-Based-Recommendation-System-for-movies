import pandas as pd
import numpy as np

def calculate_weighted_rating(data):
    C = data['vote_average'].mean()
    m = data['vote_count'].quantile(0.60)

    def wr_func(x):
        v = x['vote_count']
        R = x['vote_average']
        return (v / (v + m) * R) + (m / (v + m) * C)

    data['weighted_rating'] = data.apply(wr_func, axis=1)
    return data, C, m

# 1. Recommendation function for baseline model
def get_baseline_recommendations(df, n=10, min_votes=None, genre_filter=None):
    if min_votes is None:
        m = df['vote_count'].quantile(0.60)
        min_votes = m
    
    filtered_df = df[df['vote_count'] >= min_votes].copy()
    
    if genre_filter:
        filtered_df = filtered_df[
            filtered_df['genres'].apply(lambda x: genre_filter in x)
        ]
    
    top_n = filtered_df.sort_values('weighted_rating', ascending=False).head(n)
    return top_n

# 2. Recommendation function for content-based model
def recommendation(title, cosine_similarity, data_after_json, alpha=0.3, min_votes=50, use_mmr=True, lambda_mmr=0.5):
    indices = pd.Series(data_after_json.index, index=data_after_json['original_title']).drop_duplicates()

    if title not in indices:
        return pd.DataFrame() 
    
    idx = indices[title]
    if isinstance(idx, pd.Series):
        idx = idx.iloc[0]

    reference_age = data_after_json.iloc[idx]['movie_age']
    if pd.isna(reference_age):
        reference_age = data_after_json['movie_age'].median()

    similarity_scores = list(enumerate(cosine_similarity[idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similarity_scores = similarity_scores[1:201] 

    movie_indices = [i[0] for i in similarity_scores]
    similarity_values = [i[1] for i in similarity_scores]
    
    candidates = data_after_json.iloc[movie_indices].copy()
    
    candidates = candidates[candidates['vote_count'] >= min_votes]
    
    sim_map = dict(zip(movie_indices, similarity_values))
    candidates['similarity'] = candidates.index.map(sim_map)

    def normalize(series):
        min_val = series.min()
        max_val = series.max()
        if max_val - min_val == 0:
            return series.apply(lambda x: 0.5) 
        return (series - min_val) / (max_val - min_val)
    
    candidates['quality_norm'] = normalize(candidates['vote_average'])
    
    log_votes = np.log1p(candidates['vote_count'])
    max_log = np.log1p(data_after_json['vote_count'].max())
    candidates['popularity_penalty'] = 1 - (0.3 * (log_votes / max_log)) 

    candidates['similarity_norm'] = normalize(candidates['similarity'])
    
    candidates['age_diff'] = abs(candidates['movie_age'] - reference_age)
    candidates['age_pen'] = 1 / (1 + candidates['age_diff'] / 10)

    candidates['final_score'] = (
        (alpha * candidates['quality_norm']) +
        ((1 - alpha) * candidates['similarity_norm'])
    ) * candidates['age_pen'] * candidates['popularity_penalty']

    # MMR
    if use_mmr and len(candidates) > 10:
        selected_indices = []
        remaining_candidates = candidates.copy()
        
        first_idx = remaining_candidates['final_score'].idxmax()
        selected_indices.append(first_idx)
        remaining_candidates = remaining_candidates.drop(first_idx)
        
        while len(selected_indices) < 20 and len(remaining_candidates) > 0:
            current_similarities = cosine_similarity[remaining_candidates.index.to_list(), :][:, selected_indices]

            max_sim_to_selected = np.max(current_similarities, axis=1) if len(selected_indices) > 0 else 0
            
            mmr_scores = (lambda_mmr * remaining_candidates['final_score']) - \
                         ((1 - lambda_mmr) * max_sim_to_selected)
            
            best_idx = mmr_scores.idxmax()
            selected_indices.append(best_idx)
            remaining_candidates = remaining_candidates.drop(best_idx)
        
        candidates = candidates.loc[selected_indices]
    else:
        candidates = candidates.sort_values('final_score', ascending=False).head(20)

    return candidates[['original_title', 'movie_age', 'vote_average', 'vote_count', 'hybrid_score',
                       'final_score', 'similarity', 'genres']].reset_index()