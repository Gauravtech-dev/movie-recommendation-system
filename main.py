import os
import ast
import pickle
import asyncio
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

OMDB_API_KEY = os.getenv("OMDB_API_KEY")
OMDB_BASE_URL = "https://www.omdbapi.com/"

if not OMDB_API_KEY:
    raise RuntimeError("OMDB_API_KEY missing. Put it in .env as OMDB_API_KEY=xxxx")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "movies_metadata.csv")
DF_PATH = os.path.join(BASE_DIR, "df.pkl")
INDICES_PATH = os.path.join(BASE_DIR, "indices.pkl")
TFIDF_MATRIX_PATH = os.path.join(BASE_DIR, "tfidf_matrix.pkl")
TFIDF_PATH = os.path.join(BASE_DIR, "tfidf.pkl")

app = FastAPI(title="Movie Recommender API", version="5.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

movies = None
df = None
indices_obj = None
tfidf_matrix = None
tfidf_obj = None
TITLE_TO_IDX = {}

POSTER_CACHE: Dict[str, Optional[str]] = {}
DETAIL_CACHE: Dict[str, Dict[str, Any]] = {}


class MovieCard(BaseModel):
    movie_id: int
    title: str
    poster_url: Optional[str] = None
    release_date: Optional[str] = None
    vote_average: Optional[float] = None


class MovieDetails(BaseModel):
    movie_id: int
    title: str
    overview: Optional[str] = None
    release_date: Optional[str] = None
    poster_url: Optional[str] = None
    backdrop_url: Optional[str] = None
    genres: List[dict] = []
    vote_average: Optional[float] = None


class TFIDFRecItem(BaseModel):
    title: str
    score: float
    movie: Optional[MovieCard] = None


class SearchBundleResponse(BaseModel):
    query: str
    movie_details: MovieDetails
    tfidf_recommendations: List[TFIDFRecItem]
    genre_recommendations: List[MovieCard]


def normalize_title(title: str) -> str:
    return str(title).strip().lower()


def safe_float(value):
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def parse_genres(value):
    if value is None or pd.isna(value):
        return []
    try:
        data = ast.literal_eval(str(value))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def get_genre_names(value) -> List[str]:
    return [
        g.get("name")
        for g in parse_genres(value)
        if isinstance(g, dict) and g.get("name")
    ]


@app.on_event("startup")
def load_resources():
    global movies, df, indices_obj, tfidf_matrix, tfidf_obj, TITLE_TO_IDX

    if not os.path.exists(CSV_PATH):
        raise RuntimeError("movies_metadata.csv not found.")

    movies = pd.read_csv(CSV_PATH, low_memory=False)
    movies["id"] = pd.to_numeric(movies["id"], errors="coerce")
    movies = movies.dropna(subset=["id"])
    movies["id"] = movies["id"].astype(int)
    movies["title"] = movies["title"].fillna("").astype(str)
    movies["overview"] = movies["overview"].fillna("").astype(str)
    movies["release_date"] = movies["release_date"].fillna("").astype(str)

    with open(DF_PATH, "rb") as f:
        df = pickle.load(f)

    with open(INDICES_PATH, "rb") as f:
        indices_obj = pickle.load(f)

    with open(TFIDF_MATRIX_PATH, "rb") as f:
        tfidf_matrix = pickle.load(f)

    with open(TFIDF_PATH, "rb") as f:
        tfidf_obj = pickle.load(f)

    if not hasattr(indices_obj, "items"):
        raise RuntimeError("indices.pkl does not contain a title-to-index mapping.")

    TITLE_TO_IDX = {
        normalize_title(k): int(v)
        for k, v in indices_obj.items()
    }

    if df is None or "title" not in df.columns:
        raise RuntimeError("df.pkl must contain a 'title' column.")

    print(f"Loaded {len(movies)} movies from CSV")
    print("TF-IDF resources loaded successfully")


async def omdb_request(
    params: Dict[str, Any],
    allow_not_found: bool = False,
):
    query = dict(params)
    query["apikey"] = OMDB_API_KEY

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.get(OMDB_BASE_URL, params=query)

        response.raise_for_status()
        data = response.json()

        if data.get("Response") == "False":
            if allow_not_found:
                return None
            raise HTTPException(
                status_code=404,
                detail=data.get("Error", "Movie not found"),
            )

        return data

    except HTTPException:
        raise
    except httpx.RequestError as e:
        if allow_not_found:
            return None
        raise HTTPException(
            status_code=502,
            detail=f"OMDb request failed: {str(e)}",
        )
    except Exception as e:
        if allow_not_found:
            return None
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}",
        )


async def get_omdb_details(title: str):
    key = normalize_title(title)

    if key in DETAIL_CACHE:
        return DETAIL_CACHE[key]

    data = await omdb_request(
        {"t": title, "plot": "full"},
        allow_not_found=True,
    )

    if data:
        DETAIL_CACHE[key] = data

    return data


def local_tmdb_poster(title: str) -> Optional[str]:
    """Return the TMDB poster URL stored in the local metadata CSV."""
    if movies is None or "poster_path" not in movies.columns:
        return None

    key = normalize_title(title)

    exact = movies[movies["title"].str.lower() == key]

    if exact.empty:
        partial = movies[
            movies["title"].str.lower().str.contains(
                key, regex=False, na=False
            )
        ]
        if partial.empty:
            return None
        row = partial.iloc[0]
    else:
        row = exact.iloc[0]

    path = row.get("poster_path")

    if path is None or pd.isna(path):
        return None

    path = str(path).strip()

    if not path or path.lower() in {"nan", "none", "n/a"}:
        return None

    if path.startswith("http://") or path.startswith("https://"):
        return path

    if not path.startswith("/"):
        path = "/" + path

    return "https://image.tmdb.org/t/p/w500" + path


async def get_omdb_poster(title: str) -> Optional[str]:
    """
    Poster priority:
    1. Local TMDB poster_path from movies_metadata.csv
    2. Exact OMDb title lookup
    3. OMDb search fallback
    """
    key = normalize_title(title)

    if key in POSTER_CACHE:
        return POSTER_CACHE[key]

    # 1. Local TMDB poster
    local_poster = local_tmdb_poster(title)
    if local_poster:
        POSTER_CACHE[key] = local_poster
        return local_poster

    # 2. Exact OMDb title lookup
    data = await get_omdb_details(title)
    poster = data.get("Poster") if data else None

    if poster and poster != "N/A":
        POSTER_CACHE[key] = poster
        return poster

    # 3. OMDb search fallback
    search_data = await omdb_request(
        {"s": title, "type": "movie"},
        allow_not_found=True,
    )

    if search_data:
        search_results = search_data.get("Search") or []

        for item in search_results:
            result_title = item.get("Title", "")
            result_poster = item.get("Poster")

            if (
                normalize_title(result_title) == key
                and result_poster
                and result_poster != "N/A"
            ):
                POSTER_CACHE[key] = result_poster
                return result_poster

        for item in search_results:
            result_poster = item.get("Poster")
            if result_poster and result_poster != "N/A":
                POSTER_CACHE[key] = result_poster
                return result_poster

    POSTER_CACHE[key] = None
    return None


async def enrich_cards(rows) -> List[MovieCard]:
    rows = list(rows)

    if not rows:
        return []

    posters = await asyncio.gather(
        *(get_omdb_poster(str(row.title)) for row in rows)
    )

    cards = []

    for row, poster in zip(rows, posters):
        release_date = getattr(row, "release_date", "")
        vote_average = getattr(row, "vote_average", None)

        cards.append(
            MovieCard(
                movie_id=int(row.id),
                title=str(row.title),
                poster_url=poster,
                release_date=str(release_date) or None,
                vote_average=safe_float(vote_average),
            )
        )

    return cards


def find_movie_by_id(movie_id: int):
    result = movies[movies["id"] == movie_id]

    if result.empty:
        raise HTTPException(
            status_code=404,
            detail="Movie not found in local dataset.",
        )

    return result.iloc[0]


def find_movie_by_title(title: str):
    query = normalize_title(title)

    exact = movies[movies["title"].str.lower() == query]

    if not exact.empty:
        return exact.iloc[0]

    partial = movies[
        movies["title"]
        .str.lower()
        .str.contains(query, regex=False, na=False)
    ]

    if not partial.empty:
        return partial.iloc[0]

    return None


def tfidf_recommend_titles(
    query_title: str,
    top_n: int = 10,
) -> List[Tuple[str, float]]:
    if df is None or tfidf_matrix is None:
        raise HTTPException(
            status_code=500,
            detail="TF-IDF resources not loaded.",
        )

    key = normalize_title(query_title)

    if key in TITLE_TO_IDX:
        idx = TITLE_TO_IDX[key]
    else:
        matches = [
            title
            for title in TITLE_TO_IDX
            if key in title
        ]

        if not matches:
            return []

        idx = TITLE_TO_IDX[matches[0]]

    query_vector = tfidf_matrix[idx]
    scores = (tfidf_matrix @ query_vector.T).toarray().ravel()
    order = np.argsort(-scores)

    recommendations = []

    for i in order:
        if int(i) == int(idx):
            continue

        try:
            title = str(df.iloc[int(i)]["title"])
        except Exception:
            continue

        if not title.strip():
            continue

        recommendations.append(
            (title, float(scores[int(i)]))
        )

        if len(recommendations) >= top_n:
            break

    return recommendations


@app.get("/health")
def health():
    return {
        "status": "ok",
        "mode": "OMDb + Local TF-IDF",
        "movies_loaded": len(movies) if movies is not None else 0,
    }


@app.get("/home", response_model=List[MovieCard])
async def home(
    category: str = Query("popular"),
    limit: int = Query(24, ge=1, le=30),
):
    data = movies.copy()

    if category in {"popular", "trending"}:
        data = data.sort_values(
            "popularity",
            ascending=False,
        )

    elif category == "top_rated":
        data = data.sort_values(
            "vote_average",
            ascending=False,
        )

    elif category == "now_playing":
        data = data.sort_values(
            "release_date",
            ascending=False,
        )

    elif category == "upcoming":
        dates = pd.to_datetime(
            data["release_date"],
            errors="coerce",
        )
        data = data[
            dates > pd.Timestamp.today()
        ]

    else:
        raise HTTPException(
            status_code=400,
            detail="Invalid category.",
        )

    data = data[
        data["title"].str.strip() != ""
    ].head(limit)

    return await enrich_cards(
        data.itertuples(index=False)
    )


@app.get("/search")
async def movie_search(
    query: str = Query(..., min_length=1),
):
    data = await omdb_request(
        {"s": query, "type": "movie"}
    )

    results = []

    for item in data.get("Search", []):
        title = item.get("Title", "")

        local_movie = find_movie_by_title(title)
        local_id = (
            int(local_movie["id"])
            if local_movie is not None
            else 0
        )

        poster = item.get("Poster")

        if poster == "N/A":
            poster = None

        results.append(
            {
                "id": local_id,
                "title": title,
                "poster_url": poster,
                "release_date": item.get("Year", ""),
                "imdb_id": item.get("imdbID"),
            }
        )

    return {
        "results": results,
        "total_results": len(results),
    }


@app.get(
    "/movie/id/{movie_id}",
    response_model=MovieDetails,
)
async def movie_details(movie_id: int):
    row = find_movie_by_id(movie_id)
    title = str(row["title"])

    data = await get_omdb_details(title)

    if not data:
        raise HTTPException(
            status_code=404,
            detail="Movie details not found in OMDb.",
        )

    genres = []

    if data.get("Genre"):
        genres = [
            {"name": g.strip()}
            for g in data["Genre"].split(",")
        ]

    rating = None

    if data.get("imdbRating") not in {None, "N/A"}:
        rating = safe_float(
            data.get("imdbRating")
        )

    poster = data.get("Poster")

    if poster == "N/A":
        poster = None

    if not poster:
        poster = local_tmdb_poster(title)

    return MovieDetails(
        movie_id=movie_id,
        title=data.get("Title", title),
        overview=data.get("Plot"),
        release_date=data.get("Released"),
        poster_url=poster,
        backdrop_url=None,
        genres=genres,
        vote_average=rating,
    )


@app.get(
    "/recommend/genre",
    response_model=List[MovieCard],
)
async def recommend_genre(
    movie_id: int = Query(...),
    limit: int = Query(18, ge=1, le=30),
):
    selected = find_movie_by_id(movie_id)

    genres = get_genre_names(
        selected.get("genres")
    )

    if not genres:
        return []

    target_genre = genres[0]

    mask = movies["genres"].apply(
        lambda x: target_genre in get_genre_names(x)
    )

    recommendations = movies[
        mask & (movies["id"] != movie_id)
    ]

    recommendations = (
        recommendations
        .sort_values("popularity", ascending=False)
        .head(limit)
    )

    return await enrich_cards(
        recommendations.itertuples(index=False)
    )


@app.get("/recommend/tfidf")
async def recommend_tfidf(
    title: str = Query(..., min_length=1),
    top_n: int = Query(10, ge=1, le=30),
):
    recommendations = tfidf_recommend_titles(
        title,
        top_n,
    )

    return [
        {
            "title": title,
            "score": score,
        }
        for title, score in recommendations
    ]


@app.get(
    "/movie/search",
    response_model=SearchBundleResponse,
)
async def search_bundle(
    query: str = Query(..., min_length=1),
    tfidf_top_n: int = Query(12, ge=1, le=20),
    genre_limit: int = Query(12, ge=1, le=20),
):
    local_movie = find_movie_by_title(query)

    if local_movie is None:
        raise HTTPException(
            status_code=404,
            detail=f"Movie '{query}' not found in local dataset.",
        )

    movie_id = int(local_movie["id"])

    details = await movie_details(movie_id)

    tfidf_recs = tfidf_recommend_titles(
        details.title,
        tfidf_top_n,
    )

    tfidf_items = []

    for title, score in tfidf_recs:
        local = find_movie_by_title(title)

        card = None

        if local is not None:
            poster = await get_omdb_poster(title)

            card = MovieCard(
                movie_id=int(local["id"]),
                title=str(local["title"]),
                poster_url=poster,
                release_date=(
                    str(local.get("release_date", ""))
                    or None
                ),
                vote_average=safe_float(
                    local.get("vote_average")
                ),
            )

        tfidf_items.append(
            TFIDFRecItem(
                title=title,
                score=score,
                movie=card,
            )
        )

    genre_recs = await recommend_genre(
        movie_id,
        genre_limit,
    )

    return SearchBundleResponse(
        query=query,
        movie_details=details,
        tfidf_recommendations=tfidf_items,
        genre_recommendations=genre_recs,
    )
