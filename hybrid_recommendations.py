import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class HybridRecommenderSystem:
    """
    Combines content-based and collaborative filtering similarity scores
    using a dynamic weighted approach to produce hybrid recommendations.
    weight_content_based is settable per-call (e.g. via a diversity slider);
    the collaborative weight is always (1 - weight_content_based).
    """

    def __init__(self, number_of_recommendations=10, weight_content_based=0.3):
        self.k = number_of_recommendations
        self.weight_content_based = weight_content_based
        self.weight_collaborative_filtering = 1 - weight_content_based

    @staticmethod
    def calculate_content_based_similarities(song_index, transform_matrix):
        input_vector = transform_matrix[song_index].reshape(1, -1)
        return cosine_similarity(input_vector, transform_matrix).ravel()

    @staticmethod
    def calculate_collaborative_filtering_similarities(track_id, track_ids, interaction_matrix):
        ind = np.where(track_ids == track_id)[0].item()
        input_vector = interaction_matrix[ind]
        return cosine_similarity(input_vector, interaction_matrix).ravel()

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

    def give_recommendations(self, song_name, artist_name, songs_data, transform_matrix, interaction_matrix, track_ids):
        song_row = songs_data.loc[
            (songs_data["name"] == song_name) & (songs_data["artist"] == artist_name)
        ]
        if song_row.empty:
            raise ValueError("Song not found in the dataset.")

        song_index = song_row.index[0]
        track_id = song_row["track_id"].values.item()

        content_based_scores = self.calculate_content_based_similarities(song_index, transform_matrix)
        collaborative_filtering_scores = self.calculate_collaborative_filtering_similarities(
            track_id, track_ids, interaction_matrix
        )

        normalized_content_based = self.normalize_similarities(content_based_scores)
        normalized_collaborative_filtering = self.normalize_similarities(collaborative_filtering_scores)

        weighted_scores = self.calculate_weighted_combination(
            normalized_content_based, normalized_collaborative_filtering
        )

        top_k_indexes = np.argsort(weighted_scores)[-self.k - 1:][::-1]
        top_k_songs = songs_data.iloc[top_k_indexes]

        return top_k_songs[["name", "artist", "spotify_preview_url"]].reset_index(drop=True)