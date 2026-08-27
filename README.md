🌐 Live Demo

Backend API:
https://movie-recommendation-system-pdca.onrender.com

Health Check:
https://movie-recommendation-system-pdca.onrender.com/health

The backend is deployed on Render. The Streamlit frontend can be added here when its public URL is finalized.

📌 Overview

The Movie Recommendation System is a full-stack ML application designed to recommend movies based on the content and characteristics of a selected movie.

The system combines:

TF-IDF vectorization for representing movie content

Cosine similarity for content-based recommendations

Genre-based filtering for an additional recommendation strategy

OMDb API for movie search, details, ratings, and posters

FastAPI for serving the ML recommendation backend

Streamlit for the interactive frontend

The application works with a local dataset containing 45,463 movies and exposes the recommendation functionality through a production API.

✨ Key Features

🎯 Content-Based Recommendations

Uses TF-IDF representations and cosine similarity to find movies with similar content.

🎭 Genre-Based Recommendations

Finds popular movies belonging to the selected movie's genre.

🔎 Movie Search

Searches movies using the OMDb API and connects results with the local movie dataset where possible.

🎞️ Movie Details

Displays movie information including:

Title

Overview / Plot

Release information

Genres

IMDb rating

Poster

🖼️ Dynamic Posters

Movie posters are retrieved through OMDb, with the application handling unavailable poster responses safely.

⚡ FastAPI Backend

The ML logic is exposed through clean REST endpoints and can be consumed independently of the frontend.

🖥️ Streamlit Frontend

Provides an interactive interface for searching movies and exploring recommendations.

🧠 Recommendation Architecture

                    ┌──────────────────────┐
                    │   Movie Dataset      │
                    │    45,463 movies     │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  TF-IDF Vectorizer   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ TF-IDF Matrix        │
                    └──────────┬───────────┘
                               │
                    Selected Movie
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Cosine Similarity    │
                    └──────────┬───────────┘
                               │
                               ▼
                    Similar Movie Ranking

A second recommendation path uses the movie's genre:

Selected Movie
      │
      ▼
Extract Genre
      │
      ▼
Filter Local Dataset
      │
      ▼
Rank by Popularity
      │
      ▼
Genre Recommendations

🛠️ Tech Stack

Layer

Technology

Language

Python 3.10

Frontend

Streamlit

Backend

FastAPI

Server

Uvicorn

ML

scikit-learn

Data Processing

Pandas, NumPy

Similarity

Cosine Similarity

Movie Metadata

OMDb API

HTTP Client

HTTPX

Configuration

python-dotenv

Model/Data Persistence

Pickle

Deployment

Render

📂 Project Structure

movie-recommendation-system/
│
├── app.py
├── main.py
├── movies_metadata.csv
│
├── df.pkl
├── indices.pkl
├── tfidf.pkl
├── tfidf_matrix.pkl
│
├── requirements.txt
├── runtime.txt
├── .python-version
├── .env
├── .gitignore
├── Dockerfile
└── README.md

.env contains secrets locally and should never be committed to GitHub.

🔌 API Endpoints

Health Check

GET /health

Example response:

{
  "status": "ok",
  "mode": "OMDb + Local TF-IDF",
  "movies_loaded": 45463
}

Home / Movie Categories

GET /home?category=trending&limit=3

Supported categories include:

popular

trending

top_rated

now_playing

upcoming

Search

GET /search?query=Inception

Movie Details

GET /movie/id/{movie_id}

Genre Recommendations

GET /recommend/genre?movie_id={movie_id}&limit=18

TF-IDF Recommendations

GET /recommend/tfidf?title=Inception&top_n=10

Combined Movie Recommendation Response

GET /movie/search?query=Inception

This endpoint combines:

Movie details

TF-IDF recommendations

Genre recommendations

⚙️ How It Works

1. Data Loading

The application loads the local movie metadata dataset and precomputed ML resources during FastAPI startup.

2. TF-IDF Representation

Movie content is converted into numerical vectors using a TF-IDF vectorizer.

3. Similarity Calculation

For a selected movie, cosine similarity scores are calculated against the TF-IDF matrix.

Movies with the highest similarity scores are returned as content-based recommendations.

4. Genre Recommendation

The selected movie's genre is extracted and used to filter the local dataset. Results are ranked using popularity.

5. OMDb Enrichment

Movie titles are sent to OMDb when metadata, ratings, details, or posters are required.

6. Frontend

Streamlit consumes the FastAPI endpoints and presents the results through the web interface.

🚀 Run Locally

1. Clone the repository

git clone https://github.com/Gauravtech-dev/movie-recommendation-system.git
cd movie-recommendation-system

2. Create a virtual environment

Windows:

python -m venv venv
.\venv\Scripts\Activate.ps1

3. Install dependencies

pip install -r requirements.txt

4. Configure OMDb

Create a .env file:

OMDB_API_KEY=your_api_key_here

Do not commit the .env file.

5. Start the FastAPI backend

uvicorn main:app --reload --port 8000

Backend:

http://127.0.0.1:8000

Health check:

http://127.0.0.1:8000/health

6. Start Streamlit

In another terminal:

streamlit run app.py

🔐 Environment Variables

Variable

Description

OMDB_API_KEY

API key used to retrieve movie metadata and posters

For production deployment, configure the secret through the hosting platform rather than committing it to source control.

📊 Dataset & ML Resources

The application uses a local movie metadata dataset containing 45,463 movie records.

Precomputed resources are used for efficient recommendation inference:

TF-IDF vectorizer

TF-IDF matrix

Movie index mapping

Processed movie dataframe

This avoids rebuilding the recommendation representation every time the API starts serving a request.

🧪 Validation

The deployed backend has been validated with:

/status: ok
/mode: OMDb + Local TF-IDF
/movies_loaded: 45463

The application flow has also been tested across:

Home movie listings

Search

Movie details

TF-IDF recommendations

Genre recommendations

Movie posters

Frontend ↔ backend communication

🎯 Engineering Highlights

This project demonstrates practical ML engineering beyond model training:

Building a recommendation pipeline

Persisting ML artifacts for inference

Serving ML functionality through REST APIs

Integrating a third-party API

Handling missing external API data

Separating frontend and backend responsibilities

Deploying the backend to a cloud platform

Managing Python runtime compatibility

Testing an end-to-end production workflow

🔮 Future Improvements

Add user-specific recommendation history

Introduce collaborative filtering

Combine content and collaborative signals into a hybrid recommender

Add recommendation explanations

Improve caching for external API requests

Add automated API tests

Add CI/CD with GitHub Actions

Add monitoring and request analytics

👨‍💻 Author

Gaurav

GitHub:
https://github.com/Gauravtech-dev

📄 License

This project is intended for educational and portfolio purposes.