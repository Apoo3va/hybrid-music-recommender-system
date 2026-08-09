import pandas as pd
import joblib
from scipy.sparse import save_npz

from data_cleaning import data_for_content_filtering

FILTERED_DATA_PATH = "data/collab_filtered_data.csv"
TRANSFORMER_PATH = "transformer.joblib"
SAVE_PATH = "data/transformed_hybrid_data.npz"


def main():
    filtered_data = pd.read_csv(FILTERED_DATA_PATH)

    prepared_data = data_for_content_filtering(filtered_data)

    transformer = joblib.load(TRANSFORMER_PATH)
    transformed_data = transformer.transform(prepared_data)

    save_npz(SAVE_PATH, transformed_data)


if __name__ == "__main__":
    main()