import pandas as pd
import numpy as np
import ast
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import MinMaxScaler, normalize
from scipy.sparse import csr_matrix, hstack
from datetime import datetime

# JSON Parsing
def parse_json_columns(data):
    json_columns = ['genres', 'keywords', 'production_companies', 
                    'production_countries', 'spoken_languages', 'cast', 'crew']

    def safe_parse(x):
        try:
            if isinstance(x, list): return x
            if pd.isna(x) or x == '': return []
            return ast.literal_eval(x)
        except (ValueError, SyntaxError):
            return []

    for col in json_columns:
        data[col] = data[col].apply(safe_parse)

    # Extract names
    for col in ['genres', 'keywords', 'production_companies', 'production_countries', 'spoken_languages']:
        data[col] = data[col].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])

    # Top 5 Cast
    data['cast'] = data['cast'].apply(lambda x: [i['name'] for i in x[:5]] if isinstance(x, list) else [])

    # Director
    def get_director(x):
        if isinstance(x, list):
            for i in x:
                if i['job'] == 'Director':
                    return i['name']
        return ''
    
    data['director'] = data['crew'].apply(get_director)
    data = data.drop('crew', axis=1)
    
    # Clean text columns for soup (lowercase, no spaces)
    def clean_list(x):
        if isinstance(x, list): return [str.lower(i.replace(" ", "")) for i in x]
        if isinstance(x, str): return str.lower(x.replace(" ", ""))
        return ""

    for feature in ['cast', 'keywords', 'director', 'genres']:
        data[feature] = data[feature].apply(clean_list)
        
    return data

# Some operations from Feature Engineering
def add_engineered_features(data):
    current_year = datetime.now().year
    data['movie_age'] = current_year - data['release_year']
    data['log_movie_age'] = np.log1p(data['movie_age'].fillna(0))
    data['log_vote_count'] = np.log1p(data['vote_count'])
    data['log_popularity'] = np.log1p(data['popularity'])

    data.loc[data['runtime'] <= 15, 'runtime'] = np.nan
    data['runtime'] = data['runtime'].fillna(data['runtime'].median())

    scaler = MinMaxScaler()

    data[['popularity_scaled', 'vote_average_scaled']] = scaler.fit_transform(
        data[['popularity', 'vote_average']]
    )

    data['hybrid_score'] = (
        0.6 * data['vote_average_scaled'] +
        0.4 * data['popularity_scaled']
    )

    C = data['vote_average'].mean()
    m = data['vote_count'].quantile(0.60)
    
    #Weighted rating
    def WR(x):
        v = x['vote_count']
        R = x['vote_average']
        return (v / (v + m) * R) + (m / (v + m) * C)

    data['score'] = data.apply(WR, axis=1)
    scaler_score = MinMaxScaler()
    data['scaled_score'] = scaler_score.fit_transform(data[['score']])

    # Enhance Overview
    data['overview'] = data.apply(lambda x: str(x['overview']) + ' ' + str(x['tagline']) if len(str(x['overview'])) < 100 else str(x['overview']), axis=1)
    
    return data

# Soup & Matrix - from the second model
def build_matrices(data):
    # 1. Weights for genres
    all_genres = [g for genres in data['genres'] for g in genres]
    genre_counts = Counter(all_genres)
    total_movies = len(data)
    genre_weights = {g: np.log(total_movies / c) for g, c in genre_counts.items()}
    max_weight = max(genre_weights.values()) if genre_weights else 1.0

    # 2. Cooking Soup
    def cook_soup(row):
        # Cast x1, Keywords x2, Director x3
        soup = (' '.join(row['keywords']) * 2) + ' ' + ' '.join(row['cast']) + ' ' + (str(row['director']) + ' ') * 3
        # Weighted Genres
        for genre in row['genres']:
            weight = genre_weights.get(genre, 1.0)
            repeats = min(int((weight / max_weight) * 4) + 2, 5)
            soup += (genre + ' ') * repeats
        return soup

    data['soup'] = data.apply(cook_soup, axis=1)

    # 3. Vectorization
    tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1,2), max_features=5000, min_df=2, max_df=0.8)
    tfidf_matrix = tfidf.fit_transform(data['overview'])

    count = CountVectorizer(stop_words='english', max_features=3000, min_df=2)
    count_matrix = count.fit_transform(data['soup'])

    # 4. Numerical
    num_features = data[['log_movie_age', 'runtime', 'vote_average']]
    num_features = num_features.copy()

    for col in num_features.columns:
        num_features[col] = num_features[col].fillna(num_features[col].median())
        num_features_scaled = MinMaxScaler().fit_transform(num_features)
        num_features_sparse = csr_matrix(num_features_scaled)

    # Combine
    combined_matrix = normalize(hstack([tfidf_matrix * 2.5, count_matrix * 1.5, num_features_sparse * 0.2]))
    
    return combined_matrix, data