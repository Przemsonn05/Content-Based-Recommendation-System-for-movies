import pandas as pd
import os
import numpy as np

def load_and_merge_data(movies_path, credits_path):
    if not os.path.exists(movies_path) or not os.path.exists(credits_path):
        raise FileNotFoundError(f"Files not found: {movies_path} or {credits_path}")

    movies = pd.read_csv(movies_path)
    credits = pd.read_csv(credits_path)

    # Merge
    data = movies.merge(credits, left_on='id', right_on='movie_id', how='outer')
    
    # Basic cleaning
    data = data.drop(['homepage', 'id'], axis=1) 
    data['tagline'] = data['tagline'].fillna('')
    data['overview'] = data['overview'].fillna('Unknown')
    data['release_date'] = data['release_date'].fillna('Unknown')
    data['runtime'] = data['runtime'].fillna(data['runtime'].median()) # Lepsze niż 0

    # Drop duplicate title columns
    if 'title_x' in data.columns and 'title_y' in data.columns:
        if (data['title_x'] == data['title_y']).all():
            data = data.drop(['title_x', 'title_y'], axis=1)

    # Format numeric columns
    data['budget_formatted'] = data['budget'].apply(lambda x: f"{x:,}")
    data['revenue_formatted'] = data['revenue'].apply(lambda x: f"{x:,}")
    data.drop(['budget', 'revenue'], axis=1, inplace=True)

    # Format dates
    data['release_date'] = pd.to_datetime(data['release_date'], errors='coerce')
    data['release_year'] = data['release_date'].dt.year.astype('Int64')

    return data