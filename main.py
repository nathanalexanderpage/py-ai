# import libraries
import pandas as pd
import numpy as np
import os

from dotenv import load_dotenv
load_dotenv()

# constants
CSV_FILEPATH_BASE = os.getenv('CSV_FILEPATH_BASE')
CSV_USERS_FILEPATH = os.getenv('CSV_USERS_FILEPATH')
CSV_RATINGS_FILEPATH = os.getenv('CSV_RATINGS_FILEPATH')
CSV_ITEMS_FILEPATH = os.getenv('CSV_ITEMS_FILEPATH')
print('\nCSV filepaths are as follows:\n{}\n{}\n{}\n'.format(CSV_USERS_FILEPATH, CSV_RATINGS_FILEPATH, CSV_ITEMS_FILEPATH))

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

# look at user data
print('\nUser Data :')
print('shape : ', users.shape)
print(users.head())

# look at ratings data
print("\nRatings Data :")
print("shape : ", ratings.shape)
print(ratings.head())

# look at items data
print("\nItem Data :")
print("shape : ", items.shape)
print(items.head(6))

ratings_train = pd.read_csv(CSV_FILEPATH_BASE + 'ua.base', sep='\t', names=r_cols, encoding='latin-1')
ratings_test = pd.read_csv(CSV_FILEPATH_BASE + 'ua.test', sep='\t', names=r_cols, encoding='latin-1')
print(ratings_train.shape, ratings_test.shape)

