import numpy as np
import pandas as pd
import dask.dataframe as dd
from scipy.sparse import csr_matrix, save_npz

SONGS_DATA_PATH = "data/Music Info.csv"
USER_HISTORY_PATH = "data/User Listening History.csv"


def filter_songs_data(songs_df, track_ids_to_keep):
    """
    Keeps only the songs that appear in the user listening history.
    """
    filtered_songs = songs_df[songs_df["track_id"].isin(track_ids_to_keep)]
    filtered_songs.reset_index(drop=True, inplace=True)
    return filtered_songs


def create_interaction_matrix(history_df):
    """
    Builds a sparse (track x user) interaction matrix from the user listening
    history using Dask, since materializing this matrix densely would
    require roughly 60GB of RAM.

    Returns:
        sparse_matrix (scipy.sparse.csr_matrix): rows = tracks, columns = users
        track_ids (dask.dataframe.categorical.Index): track_id at each row index
    """
    history_df["playcount"] = history_df["playcount"].astype(np.float64)
    history_df = history_df.categorize(columns=["user_id", "track_id"])

    track_ids = history_df["track_id"].cat.categories
    user_ids = history_df["user_id"].cat.categories
    n_tracks = len(track_ids)
    n_users = len(user_ids)

    user_mapping = history_df["user_id"].cat.codes
    track_mapping = history_df["track_id"].cat.codes

    history_df = history_df.assign(user_idx=user_mapping, track_idx=track_mapping)

    interaction_array = (
        history_df.groupby(["track_idx", "user_idx"])["playcount"]
        .sum()
        .reset_index()
        .compute()
    )

    row_indices = interaction_array["track_idx"]
    col_indices = interaction_array["user_idx"]
    values = interaction_array["playcount"]

    sparse_matrix = csr_matrix((values, (row_indices, col_indices)), shape=(n_tracks, n_users))

    return sparse_matrix, track_ids


def main():
    songs_df = pd.read_csv(SONGS_DATA_PATH, usecols=["track_id", "name", "artist", "spotify_preview_url"])
    history_df = dd.read_csv(USER_HISTORY_PATH)

    sparse_matrix, track_ids = create_interaction_matrix(history_df)

    filtered_songs = filter_songs_data(songs_df, track_ids)
    filtered_songs.to_csv("data/collab_filtered_data.csv", index=False)

    np.save("data/track_ids.npy", track_ids.to_numpy())
    save_npz("data/interaction_matrix.npz", sparse_matrix)


if __name__ == "__main__":
    main()