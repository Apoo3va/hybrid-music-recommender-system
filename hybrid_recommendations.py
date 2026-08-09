import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class HybridRecommenderSystem:
    """
    Combines content-based and collaborative filtering similarity scores
    using a weighted approach to produce hybrid song recommendations.

    Requires the songs_data, transform_matrix (content-based, transformed
    over the collab-filtered song set) and interaction_matrix (collaborative)
    to all be row-aligned by track_id order (i.e. songs_data sorted by
    track_id, matching transform_matrix and interaction_matrix row order).
    """

    def __init__(
        self,
        songs_data,
        transform_matrix,
        interaction_matrix,
        track_ids,
        song_name,
        artist_name,
        number_of_recommendations=10,
        weight_content_based=0.3,
        weight_collaborative_filtering=0.7,
    ):
        self.songs_data = songs_data
        self.transform_matrix = transform_matrix
        self.interaction_matrix = interaction_matrix
        self.track_ids = track_ids
        self.song_name = song_name
        self.artist_name = artist_name
        self.k = number_of_recommendations
        self.weight_content_based = weight_content_based
        self.weight_collaborative_filtering = weight_collaborative_filtering

    def _get_song_index(self):
        song_row = self.songs_data.loc[
            (self.songs_data["name"] == self.song_name)
            & (self.songs_data["artist"] == self.artist_name)
        ]
        if song_row.empty:
            raise ValueError("Song not found in the dataset.")
        return song_row.index[0]

    def calculate_content_based_similarities(self, song_index):
        input_vector = self.transform_matrix[song_index].reshape(1, -1)
        return cosine_similarity(input_vector, self.transform_matrix).ravel()

    def calculate_collaborative_filtering_similarities(self, song_index):
        track_id = self.songs_data.loc[song_index, "track_id"]
        ind = np.where(self.track_ids == track_id)[0].item()
        input_vector = self.interaction_matrix[ind]
        return cosine_similarity(input_vector, self.interaction_matrix).ravel()

    @staticmethod
    def normalize_similarities(similarity_scores):
        min_score = similarity_scores.min()
        max_score = similarity_scores.max()
        return (similarity_scores - min_score) / (max_score - min_score)

    def calculate_weighted_combination(self, content_based_scores, collaborative_filtering_scores):
        return (
            self.weight_content_based * content_based_scores
            + self.weight_collaborative_filtering * collaborative_filtering_scores
        )

    def give_recommendations(self):
        song_index = self._get_song_index()

        content_based_scores = self.calculate_content_based_similarities(song_index)
        collaborative_filtering_scores = self.calculate_collaborative_filtering_similarities(song_index)

        normalized_content_based = self.normalize_similarities(content_based_scores)
        normalized_collaborative_filtering = self.normalize_similarities(collaborative_filtering_scores)

        weighted_scores = self.calculate_weighted_combination(
            normalized_content_based, normalized_collaborative_filtering
        )

        top_k_indexes = np.argsort(weighted_scores)[-self.k - 1:][::-1]
        top_k_songs = self.songs_data.iloc[top_k_indexes]

        return top_k_songs[["name", "artist", "spotify_preview_url"]].reset_index(drop=True)