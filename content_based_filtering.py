import pandas as pd
import joblib
from scipy.sparse import save_npz
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from category_encoders.count import CountEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer

DATA_PATH = "data/cleaned_data.csv"

frequency_encode_cols = ["year"]
ohe_cols = ["artist", "time_signature", "key"]
tfidf_col = "tags"
standard_scale_cols = ["duration_ms", "loudness", "tempo"]
min_max_scale_cols = [
    "danceability", "energy", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence",
]


def data_for_content_filtering(data):
    """
    Prepares the data for content-based filtering by dropping unneeded
    columns, removing duplicates, filling missing tags, and lowercasing
    the artist column.
    """
    return (
        data
        .drop_duplicates(subset=["spotify_id", "year", "duration_ms"])
        .reset_index(drop=True)
        .drop(columns=["track_id", "name", "spotify_id", "genre"])
        .fillna({"tags": "no_tags"})
        .assign(artist=lambda x: x["artist"].str.lower())
    )


def train_transformer(data):
    transformer = ColumnTransformer(transformers=[
        ("frequency_encode", CountEncoder(normalize=True, return_df=True), frequency_encode_cols),
        ("ohe", OneHotEncoder(handle_unknown="ignore"), ohe_cols),
        ("tfidf", TfidfVectorizer(max_features=85), tfidf_col),
        ("standard_scale", StandardScaler(), standard_scale_cols),
        ("min_max_scale", MinMaxScaler(), min_max_scale_cols),
    ], remainder="passthrough", n_jobs=-1)

    transformer.fit(data)
    return transformer


def main():
    data = pd.read_csv(DATA_PATH)
    filtered_data = data_for_content_filtering(data)

    transformer = train_transformer(filtered_data)
    transformed_data = transformer.transform(filtered_data)

    save_npz("data/transformed_data.npz", transformed_data)
    joblib.dump(transformer, "transformer.joblib")


if __name__ == "__main__":
    main()