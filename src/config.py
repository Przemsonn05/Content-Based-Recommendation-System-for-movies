import os
import random
import numpy as np

# Global SEED
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MOVIES_PATH = os.path.join(BASE_DIR, 'data', 'tmdb_5000_movies.csv')
CREDITS_PATH = os.path.join(BASE_DIR, 'data', 'tmdb_5000_credits.csv')