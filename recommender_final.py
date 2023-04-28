import pandas as pd
import numpy as np
from zipfile import ZipFile
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
import matplotlib.pyplot as plt
from flask import Flask

app = Flask(__name__)


ratings_file = "ml-latest-small/ratings.csv"
df = pd.read_csv(ratings_file)

user_ids = df["userId"].unique().tolist()
user2user_encoded = {x: i for i, x in enumerate(user_ids)}
userencoded2user = {i: x for i, x in enumerate(user_ids)}
movie_ids = df["movieId"].unique().tolist()
movie2movie_encoded = {x: i for i, x in enumerate(movie_ids)}
movie_encoded2movie = {i: x for i, x in enumerate(movie_ids)}
df["user"] = df["userId"].map(user2user_encoded)
df["movie"] = df["movieId"].map(movie2movie_encoded)
df["rating"] = df["rating"].values.astype(np.float32)

num_users = len(user2user_encoded)
num_movies = len(movie_encoded2movie)
min_rating = min(df["rating"])
max_rating = max(df["rating"])



df = df.sample(frac=1, random_state=42)
x = df[["user", "movie"]].values

y = df["rating"].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values

train_indices = int(0.8 * df.shape[0])
val_indices = train_indices + int(0.1 * df.shape[0])
x_train, x_val,x_test, y_train, y_val,y_test = (
    x[:train_indices],
    x[train_indices:val_indices],
    x[val_indices:],
    y[:train_indices],
    y[train_indices:val_indices],
    y[val_indices:],
)

EMBEDDING_SIZE = 50


class RecommenderNet(keras.Model):
    def __init__(self, num_users, num_movies, embedding_size, **kwargs):
        super().__init__(**kwargs)
        self.num_users = num_users
        self.num_movies = num_movies
        self.embedding_size = embedding_size
        self.user_embedding = layers.Embedding(
            num_users,
            embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6),
        )
        self.user_bias = layers.Embedding(num_users, 1)
        self.movie_embedding = layers.Embedding(
            num_movies,
            embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6),
        )
        self.movie_bias = layers.Embedding(num_movies, 1)

    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])
        movie_vector = self.movie_embedding(inputs[:, 1])
        movie_bias = self.movie_bias(inputs[:, 1])
        dot_user_movie = tf.tensordot(user_vector, movie_vector, 2)
        
        x = dot_user_movie + user_bias + movie_bias
        
        return tf.nn.sigmoid(x)


model = RecommenderNet(num_users, num_movies, EMBEDDING_SIZE)
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
)

history = model.fit(
    x=x_train,
    y=y_train,
    batch_size=64,
    epochs=3,
    verbose=1,
    validation_data=(x_val, y_val),
)

movie_df = pd.read_csv("ml-latest-small/movies.csv")
df_train = df[:train_indices]


def get_movies(df,user_id,seen_movie):
  print("User is",user_id)
  movies_watched_by_user = df_train[df_train.userId == user_id]
  movies_not_watched = movie_df[
      ~movie_df["movieId"].isin(movies_watched_by_user.movieId.values)
  ]["movieId"]

  movies_not_watched = np.random.choice(movies_not_watched, 19)
  movies_not_watched = np.append(movies_not_watched,seen_movie)
  
  movies_not_watched = list(
      set(movies_not_watched).intersection(set(movie2movie_encoded.keys()))
  )
  movies_not_watched = [[movie2movie_encoded.get(x)] for x in movies_not_watched]
  
  user_encoder = user2user_encoded.get(user_id)
  user_movie_array = np.hstack(
      ([[user_encoder]] * len(movies_not_watched), movies_not_watched)
  )
  
  ratings = model.predict(user_movie_array).flatten()
  top_ratings_indices = ratings.argsort()[-10:][::-1]
  recommended_movie_ids = [
      movie_encoded2movie.get(movies_not_watched[x][0]) for x in top_ratings_indices
  ]

  print("Showing recommendations for user: {}".format(user_id))
  
  recommended_movies = movie_df[movie_df["movieId"].isin(recommended_movie_ids)]
  
  return recommended_movies

user_id = df_train.userId.sample(1).iloc[0]



@app.route('/get_recommendations/<uid>')
def getmovie(uid):
   uid = int(uid)
   return {"Recommendations":get_movies(df_train,uid,1045).values.tolist()}

if __name__ == '__main__':
   app.run()