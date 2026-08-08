import pandas as pd

DATA_PATH = "data/Music Info.csv"


def clean_data(data):
    """
    Cleans the input DataFrame by performing the following operations:
    1. Drops the 'spotify_preview_url' column.
    2. Removes duplicate rows based on 'spotify_id', 'year', and 'duration_ms'.

    Parameters:
    data (pd.DataFrame): The input DataFrame containing the data to be cleaned.

    Returns:
    pd.DataFrame: The cleaned DataFrame.
    """
    return (
        data
        .drop(columns=["spotify_preview_url"])
        .drop_duplicates(subset=["spotify_id", "year", "duration_ms"])
    )


def main():
    data = pd.read_csv(DATA_PATH)
    cleaned_data = clean_data(data)
    cleaned_data.to_csv("data/cleaned_data.csv", index=False)


if __name__ == "__main__":
    main()