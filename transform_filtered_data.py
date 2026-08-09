import pandas as pd
import joblib

from data_cleaning import data_for_content_filtering
from content_based_filtering import transform_data, save_transform_data

FILTERED_DATA_PATH = "data/collab_filtered_data.csv"
TRANSFORMER_PATH = "transformer.joblib"
SAVE_PATH = "data/transformed_hybrid_data.npz"


def main():
    filtered_data = pd.read_csv(FILTERED_DATA_PATH)
    prepared_data = data_for_content_filtering(filtered_data)

    transformer = joblib.load(TRANSFORMER_PATH)
    transformed_data = transform_data(prepared_data, transformer)

    save_transform_data(transformed_data, SAVE_PATH)


if __name__ == "__main__":
    main()