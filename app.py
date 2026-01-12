import streamlit as st
import joblib
import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="Movies Recommendation System", layout="wide")

st.markdown("""
<style>
.stApp {background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);}
.card {
    padding:30px;border-radius:15px;background:rgba(255,255,255,0.08);
    text-align:center;transition:0.3s;margin-bottom:20px;
}
.card:hover {transform:translateY(-4px);background:rgba(255,255,255,0.18);}
h1,h2,h3,h4,p,label,.stMarkdown {color:white!important;}
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
    
    candidates['final_score'] = (alpha * candidates['quality_norm']) + ((1-alpha) * candidates['similarity_norm'])

    return candidates.sort_values('final_score', ascending=False).head(10)

#Interface for the app

st.title("Movie Recommendation Sysytem")
st.markdown("Select a movie and explore recommendations")

select_movie = st.selectbox(
    "Select the movie title:",
    df['original_title'].sort_values().values,
    index=None
)

if select_movie:
    if st.button("🔍 Znajdź rekomendacje", type="primary"):
        recommendations = recommendation(select_movie, alpha=0.5)

        if recommendations is not None:
            st.success(f"Here's recommendations for the movie {select_movie}:")

            first_row = st.columns(5)
            second_row = st.columns(5)
            all_columns = first_row + second_row

            for col, (idx, movie) in zip(all_columns, recommendations.iterrows()):
                with col:
                    st.markdown(movie['original_title'])
                    genres = movie['genres']

                    if isinstance(genres, list):
                        genres_str = ', '.join(genres[:3])
                    else:
                        genres_str = str(genres).replace('[', '').replace('[]', '')

                    st.caption(f"{genres_str}")

                    st.metric(label='Rate', value=f"{movie['vote_average']}")
                    match_score = int(movie['final_score'] * 100)