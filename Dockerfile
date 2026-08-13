FROM python:3.13

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY data/cleaned_data.csv data/
COPY data/transformed_data.npz data/
COPY data/collab_filtered_data.csv data/
COPY data/interaction_matrix.npz data/
COPY data/track_ids.npy data/
COPY data/transformed_hybrid_data.npz data/

COPY app.py .
COPY data_cleaning.py .
COPY content_based_filtering.py .
COPY collaborative_filtering.py .
COPY hybrid_recommendations.py .

EXPOSE 8000

CMD ["streamlit", "run", "app.py", "--server.port=8000", "--server.address=0.0.0.0"]