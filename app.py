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
        filename="recommendation_intelligent_model.joblib"
    )
    with open(model_path, 'rb') as file:
        model = joblib.load(file)
    return model

data = load_model()

@st.cache_resource
def load_baseline_model():
    model_path = hf_hub_download(
        repo_id="Przemsonn/Recommendation_System",
        filename="recommendation_baseline_model.joblib"
    )
    with open(model_path, 'rb') as file:
        model = joblib.load(file)
    return model

data_baseline = load_baseline_model()

if 'page' not in st.session_state:
    st.session_state.page = 'home'

def go(page):
    st.session_state.page = page
    st.rerun()

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

#Baseline model

movies_df = data_baseline['movies_df']
C = data_baseline['C']
m = data_baseline['m']

def get_baseline_recommendations(df, n=10, min_votes=None, genre_filter=None):
    if min_votes is None:
        min_votes = m
    
    filtered_df = df[df['vote_count'] >= min_votes].copy()
    
    if genre_filter:
        filtered_df = filtered_df[
            filtered_df['genres'].apply(lambda x: genre_filter in x)
        ]
    
    top_n = filtered_df.sort_values('weighted_rating', ascending=False).head(n)
    return top_n

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

    #Language filtering as I mentioned in the EDA section
    if movie_language != 'en':
        bonus = np.where(candidates['original_language'] == movie_language, 0.1, 0.0)
        candidates['similarity_norm'] += bonus
        candidates['similarity_norm'] = candidates['similarity_norm'].clip(upper=1.0)
    
    candidates['final_score'] = (alpha * candidates['quality_norm']) + ((1-alpha) * candidates['similarity_norm'])

    return candidates.sort_values('final_score', ascending=False).head(10)

#Interface for the app

def home():
    st.markdown("""
    <h1 style='text-align:center;
                font-weight:800;
                letter-spacing:1px;
                margin-bottom:30px'>
        🎬 Movie Recommendation System
    </h1>
    """, unsafe_allow_html=True)

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class='card'>
            <h3>🎯 Personalized Recommendations</h3>
            <p>Discover movies similar to the ones you already love</p>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Explore movies", use_container_width=True):
            go("search")

    with col2:
        st.markdown("""
        <div class='card'>
            <h3>🔥 Popular Movies — Curated Picks</h3>
            <p>See what’s trending and highly rated right now</p>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Browse top movies", use_container_width=True):
            go("look")

    st.divider()

    st.markdown("<h2 style='text-align:center;'>🎬 About the Project</h2>", unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center;">

    This project is a **movie recommendation system** designed to help users discover films they are likely to enjoy.
    It combines **data-driven popularity curation** with **semantic, content-based recommendations** powered by NLP.

    The system solves two key problems:
                
    **Cold Start** – when no user preferences are known
            
    **Personalized Discovery** – when the user selects a movie they like

    ---

    ### 🧊 Baseline Model – Cold Start Solution

    The **Baseline Model** is used when the user has no prior interaction history.
    Instead of relying on raw averages, it applies a **Weighted Rating formula**, which balances:
                
     movie ratings
                
     number of votes (confidence)
                
     overall dataset statistics
                

    ✔ Prevents highly-rated but obscure movies from dominating  
    ✔ Surfaces popular, trustworthy titles  
    ✔ Acts as a high-quality safety net for new users  

    The output is a list of **globally popular and reliable movies**, optionally filtered by genre.

    ---

    ### 🧠 Main Recommendation Engine – Content-Based NLP Model

    The main recommender uses a **content-based approach** enhanced with **Natural Language Processing**.

    Each movie is represented using a **metadata soup**, which includes:
    genres,
    keywords,
    cast & crew,
    additional descriptive features

    These features are vectorized using:
                
     **TF-IDF** (to emphasize unique descriptors)
                
     **Count Vectorization**
                
     **Cosine Similarity** to measure semantic closeness

    To balance relevance and variety, the system applies: **MMR (Maximal Marginal Relevance)** concepts
    and a weighted scoring function combining similarity and quality

    ✔ Finds semantically similar movies  
    ✔ Avoids repetitive recommendations  
    ✔ Adapts to user taste without popularity bias  

    ---

    ### 📊 Evaluation & Design Philosophy

    The models were evaluated using multiple metrics:
                
     **Quality** – average rating of recommendations
                
     **Diversity** – how different the recommendations are
                
     **Genre Overlap** – topical consistency
                
     **Popularity Bias** – balance between mainstream and niche content

    The final system strikes a **“Goldilocks balance”**:
    not too popular, not too obscure — just relevant.

    ---

    ### ✨ Summary

    This project demonstrates how simple statistical models and advanced Natural Language 
    Processing (NLP) techniques can work together to create a robust, explainable, 
    and user-friendly movie recommendation system.
                    
    By combining a popularity-based baseline model with a content-based similarity approach, 
    the system balances recommendation quality, diversity, and interpretability. 
    Statistical weighting ensures reliable rankings for widely rated movies, 
    while NLP-driven feature extraction and cosine similarity enable personalized 
    recommendations based on movie content.
                    
    The project emphasizes model transparency, practical performance, and real-world 
    usability, showing that effective recommendation systems do not require complex 
    deep learning architectures. The final solution is delivered as an interactive 
    Streamlit application, allowing users to explore recommendations intuitively while 
    maintaining full insight into how results are generated.

    </div>
    """, unsafe_allow_html=True)

    
def search():
    if st.button("⬅ Back"):
        go("home")

    st.subheader("🎥 Find similar movies")

    select_movie = st.selectbox(
        "Choose a movie:",
        df['original_title'].sort_values().values,
        index=None
    )

    if select_movie and st.button("🔍 Get recommendations", type="primary"):
        recommendations = recommendation(select_movie, alpha=0.5)

        if isinstance(recommendations, pd.DataFrame):
            st.success(f"Movies similar to **{select_movie}**:")

            rows = st.columns(5) + st.columns(5)

            for idx, (_, movie) in enumerate(recommendations.iterrows()):
                with rows[idx]:
                    st.markdown(
                        f'<div class="movie-title">{movie["original_title"]}</div>',
                        unsafe_allow_html=True
                    )

                    poster_url = fetch_poster(movie["original_title"])
                    st.image(poster_url, use_container_width=True)

                    genres = movie["genres"]
                    genres_str = ", ".join(genres[:3]) if isinstance(genres, list) else str(genres)

                    st.caption(genres_str)
                    st.metric("Rating", f"{movie['vote_average']:.1f}/10")

                    match = int(movie["final_score"] * 100)
                    st.progress(match, text=f"Match score: {match}%")


    print(df.columns)

def look():
    if st.button("← Back"): go("home")

    st.title("Baseline Movie Recommender 🎬")

    genre = st.selectbox("Choose a genre (optional)", options=["All"] + sorted(set(g for genres in movies_df['genres'] for g in genres)))
    num_of_movies = st.slider('Number of recommendations', 1, 20, 10)
    
    if genre == 'All':
        genre_filter = None
    else:
        genre_filter = genre

    top_recs = get_baseline_recommendations(movies_df, n=num_of_movies, genre_filter=genre_filter)

    st.subheader('Recommended Movies')
    st.dataframe(top_recs[['original_title','release_year','vote_average','vote_count','weighted_rating']])

if st.session_state.page == 'home':
    home()
if st.session_state.page == 'search':
    search()
if st.session_state.page == 'look':
    look()