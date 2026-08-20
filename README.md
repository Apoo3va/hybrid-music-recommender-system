# 🎵 Hybrid Music Recommender System

A production-grade hybrid music recommendation engine that blends **content-based filtering** and **collaborative filtering** to deliver both personalized and diverse song recommendations  built end-to-end with a versioned data pipeline, automated CI/CD, containerized deployment, and live auto-scaling infrastructure on AWS.

Live demo (when infrastructure is running): served behind an AWS Application Load Balancer.

---

## Overview

Music platforms face two competing goals:

- **User engagement** - served by *personalized* recommendations (songs that sound like what you already like)
- **User retention** - served by *diverse* recommendations (songs you wouldn't have found yourself)

This project solves both by combining two recommendation strategies into one **weighted hybrid system**, with a user-controllable diversity slider that lets the balance shift in real time.

| Approach | Solves | How |
|---|---|---|
| **Content-Based Filtering** | Personalization | Vectorizes song metadata/audio features (tempo, danceability, tags, key, etc.) and recommends by cosine similarity |
| **Collaborative Filtering** | Diversity | Builds a sparse user–item interaction matrix from listening history and recommends by item-item similarity |
| **Hybrid** | Both | `score = w₁ · content_similarity + w₂ · collaborative_similarity`, with normalized scores and a dynamic weight controlled by the user |

---

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌────────────────────┐
│  Music Info.csv  │     │ Listening History │     │   Raw Datasets      │
│   (50k songs)    │     │  (~10M records)   │     │   (DVC-tracked,     │
└────────┬─────────┘     └────────┬──────────┘     │   stored on S3)     │
         │                        │                └────────────────────┘
         ▼                        ▼
 ┌───────────────┐        ┌──────────────────────┐
 │ data_cleaning │        │ collaborative_        │
 │      │        │        │ filtering.py           │
 │      ▼        │        │ (Dask-powered,         │
 │ cleaned_data   │        │  sparse interaction    │
 └───────┬───────┘        │  matrix, ~30MB)         │
         │                └───────────┬────────────┘
         ▼                            │
 ┌───────────────────┐                ▼
 │ content_based_     │      ┌──────────────────────┐
 │ filtering.py        │      │ transform_filtered_   │
 │ (TF-IDF, one-hot,   │      │ data.py                │
 │  scaling → sparse   │      │ (re-vectorizes 30k    │
 │  matrix)             │      │  subset for hybrid)   │
 └──────────┬──────────┘      └───────────┬──────────┘
            │                             │
            └──────────────┬──────────────┘
                            ▼
                  ┌────────────────────────┐
                  │ hybrid_recommendations  │
                  │ .py                      │
                  │ (weighted combination,   │
                  │  min-max normalization,  │
                  │  cold-start handling)    │
                  └────────────┬────────────┘
                               ▼
                     ┌───────────────────┐
                     │     app.py         │
                     │  (Streamlit UI)     │
                     └───────────────────┘
```

The entire pipeline above is version-controlled with **DVC**, so every stage is reproducible with a single `dvc repro`, and all artifacts (cleaned data, sparse matrices, the trained transformer) are pulled from remote storage rather than committed to Git.

---

## Features

- **Personalized recommendations** via content-based filtering on song attributes
- **Diverse recommendations** via collaborative filtering on real listening behavior
- **Hybrid mode** with a live diversity slider (content-based ↔ collaborative weighting)
- **Cold-start handling** — songs without listening history automatically fall back to content-based recommendations instead of failing
- **Visual weight breakdown** — a bar chart showing the personalized/diverse split for the current recommendation
- **Session-state caching** — datasets load once per session, not on every interaction
- Fully automated **CI** (tests every push) and **CD** (builds & ships Docker images) via GitHub Actions
- **Auto-scaling, load-balanced production deployment** on AWS

---

## Tech Stack

| Layer | Tools |
|---|---|
| **Data & ML** | pandas, NumPy, scikit-learn, SciPy (sparse matrices), category_encoders |
| **Big data processing** | Dask (chunked processing of the ~10M-row interaction dataset) |
| **Pipeline versioning** | DVC (data + model artifact versioning), AWS S3 (remote storage) |
| **App** | Streamlit |
| **CI/CD** | GitHub Actions |
| **Containerization** | Docker |
| **Cloud infrastructure** | AWS EC2, Auto Scaling Groups, Application Load Balancer, ECR, IAM, S3 |
| **Testing** | pytest |

---

## 📁 Project Structure

```
.
├── data_cleaning.py              # Stage 1: raw data → cleaned data
├── content_based_filtering.py    # Stage 2: content-based vectorization + recommend()
├── collaborative_filtering.py    # Stage 3: interaction matrix (Dask) + recommend()
├── transform_filtered_data.py    # Stage 4: hybrid-aligned vectorization
├── hybrid_recommendations.py     # HybridRecommenderSystem: weighted combination
├── app.py                        # Streamlit application
├── test_app.py                   # CI smoke test (checks app returns HTTP 200)
├── dvc.yaml / dvc.lock           # Pipeline definition & tracked artifact hashes
├── requirements.txt              # Exact pinned dependencies
├── Dockerfile                    # Container build for deployment
├── appspec.yml, deploy/          # CodeDeploy config (see Deployment notes below)
├── .github/workflows/cicd.yaml   # CI + CD pipeline
├── notebook/                     # Exploratory notebooks (EDA, prototyping)
└── data/                         # DVC-tracked data (not committed to Git)
```

---

## Getting Started

### Prerequisites

- Python 3.13
- [DVC](https://dvc.org/) with S3 support
- AWS credentials with access to the project's S3 remote (for pulling data)

### Setup

```bash
git clone https://github.com/Apoo3va/hybrid-music-recommender-system.git
cd hybrid-music-recommender-system

python -m venv venv
venv\Scripts\Activate.ps1      # Windows
# source venv/bin/activate     # macOS/Linux

pip install -r requirements.txt

# Pull the DVC-tracked data and model artifacts
dvc pull
```

### Run the pipeline

```bash
dvc repro
```

This runs all four pipeline stages (`data_cleaning` → `transform_data` → `interaction_data` → `transformed_filtered_data`) and regenerates every artifact, skipping any stage whose inputs haven't changed.

### Run the app locally

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## CI/CD Pipeline

Every push to `main` triggers:

1. **CI** — checks out code, installs dependencies, pulls data via DVC, starts the Streamlit app, and runs `pytest` against it to confirm it's serving (HTTP 200)
2. **CD** — on CI success, builds a Docker image containing the app code and the small pre-computed data artifacts, and pushes it to Amazon ECR tagged `latest`

```yaml
# .github/workflows/cicd.yaml (simplified)
jobs:
  ci:
    steps: [checkout, setup-python, install deps, dvc pull, run app, pytest]
  cd:
    needs: ci
    steps: [checkout, dvc pull, ecr-login, docker build, docker push]
```

---

## Deployment

The application runs on an **AWS Auto Scaling Group** behind an **Application Load Balancer**:

- A custom **Launch Template** defines every new EC2 instance: it installs Docker and the AWS CLI, provisions swap space, authenticates to ECR using its attached IAM role (no stored credentials), and automatically pulls and runs the latest container image on boot.
- The **Auto Scaling Group** maintains 1–3 instances based on CPU utilization (target: 50%).
- New application versions are rolled out using the Auto Scaling Group's **Instance Refresh** feature, which replaces instances with health-checked new ones with zero manual intervention.

> **Note on CodeDeploy:** The pipeline includes `appspec.yml` and deployment scripts for AWS CodeDeploy blue-green deployment, matching a more traditional AWS deployment pattern. These are currently unused — CodeDeploy access was blocked by a pending AWS account payment verification. The Launch Template + Instance Refresh approach above is a functional, automated equivalent that's fully live in production.

---

## How the Hybrid Recommendation Works

1. **User provides** a song name and artist.
2. **Lookup**: is the song in the ~30,000-track subset with listening history?
   - **Yes** → run the full hybrid pipeline.
   - **No, but it's in the full 50,000-track catalog** → cold-start fallback to content-based only.
   - **Not found anywhere** → inform the user.
3. **Content-based score**: fetch the song's vector from the pre-computed transform matrix, compute cosine similarity against all other songs.
4. **Collaborative score**: fetch the song's row from the sparse interaction matrix (via its integer-encoded track index), compute cosine similarity against all other songs.
5. **Normalize** both score arrays (min-max scaling) so they're on a comparable scale — collaborative filtering scores are naturally much smaller due to sparsity.
6. **Combine**: `weighted_score = w₁ · content_score + w₂ · collaborative_score`, where `w₁ = 1 - (diversity / 10)` and `w₂ = diversity / 10`, set live by the user's diversity slider.
7. **Sort and return** the top-k songs.

---

## Data

- **Music Info** (~50,000 tracks): song metadata and audio features (danceability, energy, tempo, key, tags, etc.)
- **User Listening History** (~10 million records): `track_id`, `user_id`, `playcount` — used to build the collaborative filtering interaction matrix

Both datasets are DVC-tracked and stored in S3; only pointer files live in this Git repository.

---

## Roadmap / Known Limitations

- CodeDeploy blue-green deployment is configured but not currently active (see Deployment notes)
- No HTTPS/TLS on the load balancer — song preview URLs (HTTPS) can occasionally fail to load over the app's HTTP connection
- Recommendation quality is not evaluated against ground-truth metrics (precision@k, recall@k) since no labeled relevance data is available for this dataset

---

## License

See [LICENSE](./LICENSE) for details.

## Author

Apoorva Yadav
