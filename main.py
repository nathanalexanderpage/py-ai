# import libraries
import numpy as np
import os
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances 

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
print('\nratings_train.shape, ratings_test.shape')
print(ratings_train.shape, ratings_test.shape)

# find number of unique users, items
n_users = ratings.user_id.unique().shape[0]
n_items = ratings.movie_id.unique().shape[0]
print('\nn_users, n_items')
print(n_users, n_items)

# create user-item matrix
data_matrix = np.zeros((n_users, n_items))
for line in ratings.itertuples():
    data_matrix[line[1]-1, line[2]-1] = line[3]
print('\ndata_matrix')
print(data_matrix)

user_similarity = pairwise_distances(data_matrix, metric='cosine')
item_similarity = pairwise_distances(data_matrix.T, metric='cosine')

def predict(ratings, similarity, type='user'):
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        # np.newaxis - mean_user_rating has same format as ratings
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred

user_prediction = predict(data_matrix, user_similarity, type='user')
item_prediction = predict(data_matrix, item_similarity, type='item')


print('\nuser_prediction')
print(user_prediction)
print('\nitem_prediction')
print(item_prediction)