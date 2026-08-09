import numpy as np
import pandas as pd
import joblib
from scipy.sparse import save_npz
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from category_encoders.count import CountEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.metrics.pairwise import cosine_similarity

from data_cleaning import data_for_content_filtering

DATA_PATH = "data/cleaned_data.csv"
TRANSFORMED_DATA_SAVE_PATH = "data/transformed_data.npz"

frequency_encode_cols = ["year"]
ohe_cols = ["artist", "time_signature", "key"]
tfidf_col = "tags"
standard_scale_cols = ["duration_ms", "loudness", "tempo"]
min_max_scale_cols = [
    "danceability", "energy", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence",
]


def train_transformer(data):
    transformer = ColumnTransformer(transformers=[
        ("frequency_encode", CountEncoder(normalize=True, return_df=True), frequency_encode_cols),
        ("ohe", OneHotEncoder(handle_unknown="ignore"), ohe_cols),
        ("tfidf", TfidfVectorizer(max_features=85), tfidf_col),
        ("standard_scale", StandardScaler(), standard_scale_cols),
        ("min_max_scale", MinMaxScaler(), min_max_scale_cols),
    ], remainder="passthrough", n_jobs=-1)

    transformer.fit(data)
    joblib.dump(transformer, "transformer.joblib")
    return transformer


def transform_data(data, transformer):
    """
    Applies an already-fitted transformer to the given data.
    Reused by transform_filtered_data.py for the hybrid stage.
    """
    return transformer.transform(data)


def save_transform_data(transformed_data, save_path):
    """
    Saves a transformed (sparse) matrix to disk in .npz format.
    Reused by transform_filtered_data.py for the hybrid stage.
    """
    save_npz(save_path, transformed_data)


def recommend(song_name, songs_data, transformed_data, k=10):
    """
    Recommends the top k songs most similar to the given song, based on the
    content-based (cosine similarity) transformed feature matrix.
    """
    song_row = songs_data.loc[songs_data["name"] == song_name, :]
    if song_row.empty:
        print("Song not found in the dataset.")
        return None

    song_index = song_row.index[0]
    input_vector = transformed_data[song_index].reshape(1, -1)
    similarity_scores = cosine_similarity(input_vector, transformed_data)
    top_k_songs_indexes = np.argsort(similarity_scores.ravel())[-k - 1:][::-1]
    top_k_songs = songs_data.iloc[top_k_songs_indexes]
    top_k_list = top_k_songs[["name", "artist"]].reset_index(drop=True)
    return top_k_list


def main():
    data = pd.read_csv(DATA_PATH)
    filtered_data = data_for_content_filtering(data)

    transformer = train_transformer(filtered_data)
    transformed_data = transform_data(filtered_data, transformer)

    save_transform_data(transformed_data, TRANSFORMED_DATA_SAVE_PATH)


if __name__ == "__main__":
    main()