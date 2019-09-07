# import libraries
import pandas as pd
import numpy as np
import os

from dotenv import load_dotenv
load_dotenv()

# constants
CSV_USERS_FILEPATH = os.getenv('CSV_USERS_FILEPATH')
CSV_RATINGS_FILEPATH = os.getenv('CSV_RATINGS_FILEPATH')
CSV_ITEMS_FILEPATH = os.getenv('CSV_ITEMS_FILEPATH')

# read user file
u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv(CSV_USERS_FILEPATH, sep='|', names=u_cols,encoding='latin-1')

# read ratings file
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv(CSV_RATINGS_FILEPATH, sep='\t', names=r_cols,encoding='latin-1')

# read items file
i_cols = ['movie id', 'movie title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
items = pd.read_csv(CSV_ITEMS_FILEPATH, sep='|', names=i_cols,
encoding='latin-1')