# 🎬 Content-Based Movie Recommendation System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange?style=for-the-badge&logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)

</div>

## 📌 Project Overview

In the era of streaming wars, users often face **"analysis paralysis"** due to content overload. This project implements a robust movie recommendation engine designed to surface relevant content by analyzing metadata semantic similarities.

Unlike simple tag-matching systems, this engine employs a **Hybrid Logic** that balances semantic relevance, movie quality, popularity trends, and release recency. The result is a personalized discovery experience that mitigates common pitfalls of standard recommendation algorithms.

---

## 🚀 Live Demo & Resources

| Platform | Link | Description |
| :--- | :--- | :--- |
| **Streamlit App** | [![Streamlit](https://img.shields.io/badge/Launch-App-FF4B4B)](https://content-based-recommendation-system-for-movies-mr9raxrqxyopdb9.streamlit.app) | Explore the interactive dashboard and get recommendations. |
| **Hugging Face** | [![Hugging Face](https://img.shields.io/badge/Models-Registry-yellow)](https://huggingface.co/Przemsonn/Recommendation_System) | Download the trained models (hosted externally due to size). |

---

## ⚙️ Methodology & Data Pipeline

### 1. Data Cleaning & Parsing
Data precision was paramount. We performed rigorous cleaning, including parsing complex stringified lists (e.g., converting `['Science', 'Fiction']` to `sciencefiction`). This "sanitization" step was crucial for the NLP model to treat multi-word genres and names as unique tokens, preventing semantic drift.

**Key operations included:**
* **Imputation:** Missing values were filled using statistical reasoning (e.g., using medians for numerical gaps).
* **Pruning:** Irrelevant features that added noise were removed.
* **Extraction Logic:** Custom functions were built, such as `get_boss` (extracting the director) and `get_5_cat` (extracting the top 5 billed actors).

### 2. Exploratory Data Analysis (EDA)
EDA served as a decision-making tool rather than just visualization. It guided our feature selection and transformation strategies.

![Distribution of Features](images/Distribution_of_Movie_Features.png)

**Key Insights:**
* **Vote Average (Upper Left):** Most movies fall between 2 and 8. Extreme values (0 or 10) often indicated data scarcity rather than true quality.
* **Popularity (Upper Right):** A classic "Long Tail" distribution. Most scores are between 0–40, necessitating logarithmic scaling to reduce skew.
* **Runtime (Bottom Left):** The median runtime is 90–120 minutes. Outliers (<15 mins) were flagged as "shorts" or errors and filtered out to preserve model integrity.
* **Vote Count (Bottom Right):** A highly skewed metric. High vote counts correlate with blockbuster status, requiring normalization to give hidden gems a chance.

### 3. Feature Engineering
We engineered custom features to enhance the recommendation logic:

* **`weighted_rating`:** A Bayesian average balancing high ratings with vote counts. This prevents niche films with a single 10/10 rating from dominating the leaderboard.
* **`movie_age`:** Calculated to slightly penalize outdated content, balancing nostalgia with relevance.
* **`tagline_integration`:** Enriched textual data by appending taglines to overviews, increasing the semantic signal for the NLP engine.

**The Quality Score:**
To quantify "good" content, we defined a weighted score:

$$\text{Quality Score} = 0.6 \times \text{Rating} + 0.4 \times \text{Popularity}$$

---

## 🧠 Model Architecture

### 🛡️ Baseline Model: The "Cold Start" Solution
Every recommendation system faces the **Cold Start problem**: *How to serve a user with no history?*

To solve this, I built a curatorial Baseline Model. Since simple averages are misleading, I applied a **Weighted Quality Score** to penalize low-confidence ratings. This ensures default recommendations are statistically significant hits—mostly Action and Drama titles from the modern era.

**Performance:**
The chart below confirms the model's effectiveness. Recommendations (red dots) cluster in the upper-right quadrant, representing the ideal intersection of **High Popularity** and **High Quality**.

![Popularity vs Quality](images/Popularity_vs_Quality_Top10_Baseline_Recommendations.png)

### 🚀 Main Model: The Hybrid Engine
Moving from static curation to dynamic retrieval, the Main Model uses **Natural Language Processing (NLP)**.

1.  **Metadata Soup:** A composite feature vector merging keywords, cast, director, and genres.
2.  **TF-IDF Vectorization:** We used Term Frequency-Inverse Document Frequency to downweight generic terms (like "Action") and highlight unique descriptors, allowing the model to distinguish between "Space Horror" and "Space Comedy."

**The Hybrid Logic Formula:**
The final ranking is determined by re-ranking candidates using a weighted formula:

$$\text{Score} = (W_{sim} \cdot \text{Sim}) + (W_{qual} \cdot \text{Qual}) + (W_{pop} \cdot \text{Pop}) - (W_{age} \cdot \text{Age})$$

* **Similarity (50%):** Semantic relevance based on the Metadata Soup.
* **Quality (30%):** Ensures recommendations are critically acclaimed.
* **Popularity (10%):** Log-transformed to surface hidden gems.
* **Age (-10%):** A slight penalty for age to keep recommendations fresh.

---

## 📊 Performance & Evaluation

The model was stress-tested using a Monte Carlo simulation (50 samples, 20 iterations) to measure global health metrics.

| Metric | Result | Interpretation |
| :--- | :--- | :--- |
| **Quality (Avg Rating)** | **6.60 / 10** | **High Quality.** The model successfully filters out low-rated content (Global avg ~6.0). |
| **Diversity Index** | **0.72** | **High Exploration.** Recommendations are varied, avoiding "filter bubbles." |
| **Genre Overlap** | **0.64** | **Solid Consistency.** 64% of recommendations share the primary genre, ensuring thematic relevance. |
| **Popularity Bias** | **1.51** | **Goldilocks Zone.** Recommendations are 1.5x more popular than average—recognizable but not just blockbusters. |

![Metrics Visualization](images/Plot_Results_for_Second_Model.png)

**Deep Dive:**
* **Quality Analysis:** The distribution (green area) shifts right compared to the global average (red line), proving the model favors higher-rated films.
* **Diversity:** With a peak around 0.75, the system avoids homogeneity. Users won't just see 10 identical sequels.
* **Genre Overlap:** The median of 0.6 demonstrates that the model respects the user's thematic intent while allowing for adjacent genre discovery.
* **Popularity Strategy:** The bias is heavily skewed towards low-to-mid popularity (long tail), confirming the system is capable of surfacing "hidden gems" rather than just the top 100 movies of all time.

---

## 📲 App Interface & Usage

The project is deployed as an interactive web application.

**1. Movie Selection**
Users begin by selecting a movie from a database of ~4800 titles.
![Interface 1](images/streamlit_interface1.png)

**2. Recommendation Engine**
The app displays 10 semantic recommendations. For each movie, you see:
* **Match %:** The hybrid score confidence.
* **Rating:** The weighted IMDB score.
* **Genres:** Top tags for context.
* **Poster:** Fetched dynamically via the TMDB API.

![Interface 2](images/streamlit_interface2.png)
![Interface 3](images/streamlit_interface4.png)

---

## 💡 Key Learnings & Future Work

**Learnings:**
* **NLP Power:** The "Metadata Soup" combined with TF-IDF proved significantly more powerful than simple genre matching.
* **Feature Weighting:** The success of the model relied heavily on the manual tuning of weights ($W_{sim}$, $W_{qual}$, etc.).
* **Data Hygiene:** Rigorous parsing during the cleaning phase prevented duplicate tokens and improved model precision.

**Future Improvements:**
* **GenAI Integration:** Upgrade the NLP layer using **BERT** or **LLMs** (e.g., Llama/GPT) for deep semantic understanding beyond keyword matching.
* **Chatbot Interface:** Add a conversational layer to handle natural queries (e.g., "I want a sad movie about robots").
* **Dockerization:** Containerize the application for easier deployment and scalability.

---

## 🛠️ Installation

```bash
# 1. Clone the repository
git clone [https://github.com/your-username/movie-recommender.git](https://github.com/your-username/movie-recommender.git)

# 2. Navigate to directory
cd movie-recommender

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the analysis notebook (optional)
jupyter notebook Recommendation_System.ipynb

# 5. Run the Streamlit App locally
streamlit run app.py