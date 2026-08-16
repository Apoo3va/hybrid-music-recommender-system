import numpy as np
import pandas as pd
import streamlit as st
from scipy.sparse import load_npz

from content_based_filtering import recommend as content_based_recommender
from hybrid_recommendations import HybridRecommenderSystem


def load_data():
    """
    Loads all datasets once per session and caches them in st.session_state,
    so Streamlit's rerun-on-every-click behavior doesn't reload everything
    each time the user clicks "Get Recommendations".
    """
    if "content_songs_data" not in st.session_state:
        cleaned_data = pd.read_csv("data/cleaned_data.csv")
        st.session_state["content_songs_data"] = (
            cleaned_data
            .drop_duplicates(subset=["spotify_id", "year", "duration_ms"])
            .reset_index(drop=True)
        )
    if "transformed_data" not in st.session_state:
        st.session_state["transformed_data"] = load_npz("data/transformed_data.npz")
    if "collab_songs_data" not in st.session_state:
        st.session_state["collab_songs_data"] = pd.read_csv("data/collab_filtered_data.csv")
    if "interaction_matrix" not in st.session_state:
        st.session_state["interaction_matrix"] = load_npz("data/interaction_matrix.npz")
    if "track_ids" not in st.session_state:
        st.session_state["track_ids"] = np.load("data/track_ids.npy", allow_pickle=True)
    if "transformed_hybrid_data" not in st.session_state:
        st.session_state["transformed_hybrid_data"] = load_npz("data/transformed_hybrid_data.npz")


load_data()

content_songs_data = st.session_state["content_songs_data"]
transformed_data = st.session_state["transformed_data"]
collab_songs_data = st.session_state["collab_songs_data"]
interaction_matrix = st.session_state["interaction_matrix"]
track_ids = st.session_state["track_ids"]
transformed_hybrid_data = st.session_state["transformed_hybrid_data"]


def find_song(songs_data, song_name, artist_name=None):
    """
    Case-insensitive lookup that returns the actual (correctly-cased)
    name/artist values from the dataset, or None if not found.
    """
    name_mask = songs_data["name"].str.lower() == song_name.lower()
    if artist_name:
        mask = name_mask & (songs_data["artist"].str.lower() == artist_name.lower())
    else:
        mask = name_mask
    match = songs_data.loc[mask]
    if match.empty:
        return None
    row = match.iloc[0]
    return row["name"], row["artist"]


def display_recommendations(recommendations, current_song_label):
    st.write(f"Recommendations similar to '{current_song_label}':")
    for idx, row in recommendations.iterrows():
        label = "**Currently playing:**" if idx == 0 else f"{idx}."
        st.write(f"{label} {row['name']} by {row['artist']}")
        preview_url = row.get("spotify_preview_url")
        if pd.notna(preview_url):
            st.audio(preview_url)


# ---- Streamlit UI ----

st.title("Music Recommender System")
st.subheader("Enter the name of a song and get similar recommendations")

song_name = st.text_input("Song name")
artist_name = st.text_input("Artist name")

k = st.selectbox("Number of recommendations", [5, 10, 15, 20], index=1)

# Automatically decide which method to use: hybrid if the song has listening
# history (in the 30k collab subset), content-based only if it's a newer/
# niche song (cold start, only in the full 50k set), or an error if it's
# not in the database at all. No more asking the user to pick a method.
in_collab_subset = False
in_content_only = False
if song_name and artist_name:
    in_collab_subset = find_song(collab_songs_data, song_name, artist_name) is not None
if song_name:
    in_content_only = find_song(content_songs_data, song_name) is not None

weight_content_based = 0.5
if song_name and artist_name and in_collab_subset:
    diversity = st.slider(
        "Diversity (1 = more personalized, 9 = more diverse)", 1, 9, 5
    )
    weight_content_based = 1 - (diversity / 10)
    weight_collaborative = diversity / 10

    chart_data = pd.DataFrame(
        {"Weight": [weight_content_based, weight_collaborative]},
        index=["Personalized (Content-Based)", "Diverse (Collaborative)"],
    )
    st.bar_chart(chart_data)
elif song_name and in_content_only and not in_collab_subset:
    st.info(
        "This song has no listening history yet (cold start) — "
        "showing content-based (personalized) recommendations only."
    )

if st.button("Get Recommendations"):
    if not song_name or not artist_name:
        st.warning("Please enter both a song name and artist name.")
    elif in_collab_subset:
        actual_name, actual_artist = find_song(collab_songs_data, song_name, artist_name)
        recommender = HybridRecommenderSystem(
            number_of_recommendations=k,
            weight_content_based=weight_content_based,
        )
        recommendations = recommender.give_recommendations(
            actual_name, actual_artist, collab_songs_data, transformed_hybrid_data, interaction_matrix, track_ids
        )
        display_recommendations(recommendations, actual_name)
    elif in_content_only:
        actual_name, actual_artist = find_song(content_songs_data, song_name)
        recommendations = content_based_recommender(
            actual_name, content_songs_data, transformed_data, k=k
        )
        display_recommendations(recommendations, actual_name)
    else:
        st.warning("Song not found in the database. Please try another song.")