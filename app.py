import streamlit as st
import joblib
import pandas as pd
import numpy as np
import requests
import urllib.parse
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="Movies Recommendation System", layout="wide")

st.markdown("""
<style>
    .movie-title {
        height: 80px;            
        display: flex;          
        align-items: center;    
        justify-content: center; 
        text-align: center;      
        font-weight: bold;      
        font-size: 14px;       
        line-height: 1.2;       
        margin-bottom: 10px;    
        
        overflow: hidden;
        display: -webkit-box;
        -webkit-line-clamp: 3; 
        -webkit-box-orient: vertical;
        color: #fff;
    }
    
    div[data-testid="stImage"] img {
        height: 300px !important;  
        object-fit: cover !important; 
        border-radius: 8px;        
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    div[data-testid="stCaptionContainer"] {
        text-align: center;
        color: #d1d5db !important; /* Lekko srebrzysty caption */
    }

    .stApp {
        background: linear-gradient(to right, #0f0c29, #302b63, #24243e);
        background-attachment: fixed;
    }

    h1, h2, h3, h4, h5, h6, p, label, .stMarkdown {
        color: #ffffff !important;
    }

    div[data-testid="column"] {
        background-color: rgba(60, 10, 80, 0.15); /* Fioletowy odcień szkła */
        border: 1px solid rgba(200, 100, 255, 0.1); /* Delikatna fioletowa ramka */
        border-radius: 15px;      
        padding: 15px;            
        backdrop-filter: blur(10px); 
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }

    div[data-testid="column"]:hover {
        transform: translateY(-7px);
        box-shadow: 0 10px 25px rgba(130, 0, 255, 0.3); /* Neonowy cień */
        border-color: rgba(200, 100, 255, 0.4); 
    }

    .stProgress > div > div > div > div {
        background-image: linear-gradient(to right, #8E2DE2, #4A00E0);
    }
    
    div.stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border: none;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="Przemsonn/Recommendation_System",
        filename="recommendation_model.joblib"
    )

    with open(model_path, 'rb') as file:
        model = joblib.load(file)
    return model

data = load_model()

@st.cache_data
def fetch_poster(title):
    api_key = "e9b3d423b23f3b61816fe8887b777754"

    encoded_title = urllib.parse.quote(title)
    url = f"https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={encoded_title}&language=en-US"

    try:
        response = requests.get(url)
        data = response.json()

        if data['results']:
            first_result = data['results'][0]
            poster_path = first_result.get('poster_path')

            if poster_path:
                return f"https://image.tmdb.org/t/p/w500/{poster_path}"
            
        return "https://via.placeholder.com/500x750?text=No+Poster"
    
    except Exception as e:
        return "https://via.placeholder.com/500x750?text=Error"

#Recommendation function

df = data['dataframe']
cosine_sim = data['similarity']
indices = data['indices']

def recommendation(title, alpha=0.5):
    if title not in indices:
        return f"Film {title} was not found in the database"
    
    idx = indices[title]
    if isinstance(idx, pd.Series):
        idx = idx.iloc[0]

    similarity_scores = list(enumerate(cosine_sim[idx]))
    similarity_scores = sorted(similarity_scores, key = lambda x: x[1], reverse=True)
    similarity_scores = similarity_scores[1:51]

    if 'original_language' in df.columns:
        movie_language = df.loc[df['original_title'] == title, 'original_language'].iloc[0]
    else:
        movie_language = 'en'


    movie_indices = [i[0] for i in similarity_scores]
    similarity_values = [i[1] for i in similarity_scores]
    
    candidates = df.iloc[movie_indices].copy()
    candidates['similarity'] = similarity_values

    def normalize(series):
        min_val = series.min()
        max_val = series.max()
        if max_val - min_val == 0:
            return series.apply(lambda x: 0.5) 
        return (series - min_val) / (max_val - min_val)

    candidates['similarity_norm'] = normalize(candidates['similarity'])
    candidates['quality_norm'] = normalize(candidates['hybrid_score']) 

    if movie_language != 'en':
        bonus = np.where(candidates['original_language'] == movie_language, 0.1, 0.0)
        candidates['similarity_norm'] += bonus
        candidates['similarity_norm'] = candidates['similarity_norm'].clip(upper=1.0)
    
    candidates['final_score'] = (alpha * candidates['quality_norm']) + ((1-alpha) * candidates['similarity_norm'])

    return candidates.sort_values('final_score', ascending=False).head(10)

#Interface for the app

st.markdown("""
<h1 style='text-align:center;
            font-weight:800;
            letter-spacing:1px;
            margin-bottom:30px'>
    🎥 Movie Recommender
</h1>
""", unsafe_allow_html=True)

select_movie = st.selectbox(
    "Select the movie title:",
    df['original_title'].sort_values().values,
    index=None
)

if select_movie:
    if st.button("🔍 Find Recommendation", type="primary"):
        recommendations = recommendation(select_movie, alpha=0.5)

        if recommendations is not None:
            st.success(f"Here's recommendations for the movie {select_movie}:")

            first_row = st.columns(5)
            second_row = st.columns(5)
            all_columns = first_row + second_row

            for idx, (index, movie) in enumerate(recommendations.iterrows()):
                col=all_columns[idx]
                with col:
                    st.markdown(f'<div class="movie-title">{movie["original_title"]}</div>', unsafe_allow_html=True)
                    genres = movie['genres']
                    poster_url = fetch_poster(movie['original_title'])

                    st.image(poster_url, use_container_width=True)

                    if isinstance(genres, list):
                        genres_str = ', '.join(genres[:3])
                    else:
                        genres_str = str(genres).replace('[', '').replace('[]', '')

                    st.caption(f"{genres_str}")

                    st.metric(label='Rate', value=f"{movie['vote_average']:.1f}/10")
                    match_score = int(movie['final_score'] * 100)
                    st.progress(match_score, text=f"Match score: {match_score}%")

print(df.columns)