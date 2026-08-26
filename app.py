import requests
import streamlit as st

API_BASE = API_BASE = "https://movie-recommendation-system-pdca.onrender.com" or "http://127.0.0.1:8000"

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

st.markdown(
    """
<style>
.block-container { padding-top: 1rem; padding-bottom: 2rem; max-width: 1400px; }
.small-muted { color:#6b7280; font-size: 0.92rem; }
.movie-title { font-size: 0.9rem; line-height: 1.15rem; height: 2.3rem; overflow: hidden; }
.card { border: 1px solid rgba(0,0,0,0.08); border-radius: 16px; padding: 14px; background: rgba(255,255,255,0.7); }
</style>
""",
    unsafe_allow_html=True,
)

if "view" not in st.session_state:
    st.session_state.view = "home"
if "selected_movie_id" not in st.session_state:
    st.session_state.selected_movie_id = None

qp_view = st.query_params.get("view")
qp_id = st.query_params.get("id")
if qp_view in ("home", "details"):
    st.session_state.view = qp_view
if qp_id:
    try:
        st.session_state.selected_movie_id = int(qp_id)
        st.session_state.view = "details"
    except ValueError:
        pass


def goto_home():
    st.session_state.view = "home"
    st.session_state.selected_movie_id = None
    st.query_params["view"] = "home"
    if "id" in st.query_params:
        del st.query_params["id"]
    st.rerun()


def goto_details(movie_id: int):
    st.session_state.view = "details"
    st.session_state.selected_movie_id = int(movie_id)
    st.query_params["view"] = "details"
    st.query_params["id"] = str(int(movie_id))
    st.rerun()


@st.cache_data(ttl=30)
def api_get_json(path: str, params: dict | None = None):
    try:
        response = requests.get(f"{API_BASE}{path}", params=params, timeout=30)
        if response.status_code >= 400:
            return None, f"HTTP {response.status_code}: {response.text[:300]}"
        return response.json(), None
    except Exception as e:
        return None, f"Request failed: {e}"


def poster_grid(cards, cols=6, key_prefix="grid"):
    if not cards:
        st.info("No movies to show.")
        return

    rows = (len(cards) + cols - 1) // cols
    idx = 0
    for r in range(rows):
        colset = st.columns(cols)
        for c in range(cols):
            if idx >= len(cards):
                break
            movie = cards[idx]
            idx += 1

            movie_id = movie.get("movie_id") or movie.get("id")
            title = movie.get("title", "Untitled")
            poster = movie.get("poster_url")

            with colset[c]:
                if poster:
                    st.image(poster)
                else:
                    st.markdown("🖼️ **No poster**")

                if st.button("Open", key=f"{key_prefix}_{r}_{c}_{idx}_{movie_id}"):
                    if movie_id:
                        goto_details(int(movie_id))

                st.markdown(
                    f"<div class='movie-title'>{title}</div>",
                    unsafe_allow_html=True,
                )


def to_cards_from_tfidf_items(items):
    cards = []
    for item in items or []:
        movie = item.get("movie") or {}
        movie_id = movie.get("movie_id") or movie.get("id")
        if movie_id:
            cards.append(
                {
                    "movie_id": movie_id,
                    "title": movie.get("title") or item.get("title") or "Untitled",
                    "poster_url": movie.get("poster_url"),
                }
            )
    return cards


def parse_movie_search_to_cards(data, keyword: str, limit: int = 24):
    keyword_l = keyword.strip().lower()
    raw_items = []

    if isinstance(data, dict) and "results" in data:
        for item in data.get("results") or []:
            title = (item.get("title") or "").strip()
            movie_id = item.get("id") or item.get("movie_id")
            poster_url = item.get("poster_url")
            if not title or not movie_id:
                continue
            raw_items.append(
                {
                    "movie_id": int(movie_id),
                    "title": title,
                    "poster_url": poster_url,
                    "release_date": item.get("release_date") or "",
                }
            )

    elif isinstance(data, list):
        for item in data:
            title = (item.get("title") or "").strip()
            movie_id = item.get("movie_id") or item.get("id")
            poster_url = item.get("poster_url")
            if not title or not movie_id:
                continue
            raw_items.append(
                {
                    "movie_id": int(movie_id),
                    "title": title,
                    "poster_url": poster_url,
                    "release_date": item.get("release_date") or "",
                }
            )
    else:
        return [], []

    matched = [x for x in raw_items if keyword_l in x["title"].lower()]
    final_list = matched if matched else raw_items

    suggestions = []
    for item in final_list[:10]:
        year = (item.get("release_date") or "")[:4]
        label = f"{item['title']} ({year})" if year else item["title"]
        suggestions.append((label, item["movie_id"]))

    cards = final_list[:limit]
    return suggestions, cards


with st.sidebar:
    st.markdown("## 🎬 Menu")
    if st.button("🏠 Home"):
        goto_home()

    st.markdown("---")
    home_category = st.selectbox(
        "Category",
        ["trending", "popular", "top_rated", "now_playing", "upcoming"],
        index=0,
    )
    grid_cols = st.slider("Grid columns", 4, 8, 6)

st.title("🎬 Movie Recommender")
st.markdown(
    "<div class='small-muted'>Search a movie → open details → get TF-IDF and genre recommendations</div>",
    unsafe_allow_html=True,
)
st.divider()

if st.session_state.view == "home":
    typed = st.text_input(
        "Search by movie title (keyword)",
        placeholder="Type: avenger, batman, love...",
    )
    st.divider()

    if typed.strip():
        if len(typed.strip()) < 2:
            st.caption("Type at least 2 characters for suggestions.")
        else:
            data, err = api_get_json("/search", params={"query": typed.strip()})
            if err or data is None:
                st.error(f"Search failed: {err}")
            else:
                suggestions, cards = parse_movie_search_to_cards(data, typed.strip(), limit=24)
                if suggestions:
                    labels = ["-- Select a movie --"] + [s[0] for s in suggestions]
                    selected = st.selectbox("Suggestions", labels, index=0)
                    if selected != "-- Select a movie --":
                        label_to_id = {label: movie_id for label, movie_id in suggestions}
                        goto_details(label_to_id[selected])
                else:
                    st.info("No suggestions found. Try another keyword.")

                st.markdown("### Results")
                poster_grid(cards, cols=grid_cols, key_prefix="search_results")
        st.stop()

    st.markdown(f"### 🏠 Home — {home_category.replace('_', ' ').title()}")
    home_cards, err = api_get_json(
        "/home", params={"category": home_category, "limit": 24}
    )
    if err or not home_cards:
        st.error(f"Home feed failed: {err or 'Unknown error'}")
        st.stop()
    poster_grid(home_cards, cols=grid_cols, key_prefix="home_feed")

elif st.session_state.view == "details":
    movie_id = st.session_state.selected_movie_id
    if not movie_id:
        st.warning("No movie selected.")
        if st.button("← Back to Home"):
            goto_home()
        st.stop()

    a, b = st.columns([3, 1])
    with a:
        st.markdown("### 📄 Movie Details")
    with b:
        if st.button("← Back to Home"):
            goto_home()

    data, err = api_get_json(f"/movie/id/{movie_id}")
    if err or not data:
        st.error(f"Could not load details: {err or 'Unknown error'}")
        st.stop()

    left, right = st.columns([1, 2.4], gap="large")

    with left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        if data.get("poster_url"):
            st.image(data["poster_url"])
        else:
            st.write("🖼️ No poster")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"## {data.get('title', '')}")
        release = data.get("release_date") or "-"
        genres = ", ".join(g.get("name", "") for g in data.get("genres", [])) or "-"
        st.markdown(f"<div class='small-muted'>Release: {release}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='small-muted'>Genres: {genres}</div>", unsafe_allow_html=True)
        if data.get("vote_average") is not None:
            st.markdown(f"<div class='small-muted'>IMDb Rating: {data['vote_average']}</div>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("### Overview")
        st.write(data.get("overview") or "No overview available.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.divider()
    st.markdown("### ✅ Recommendations")

    title = (data.get("title") or "").strip()
    if title:
        bundle, err2 = api_get_json(
            "/movie/search",
            params={"query": title, "tfidf_top_n": 12, "genre_limit": 12},
        )

        if not err2 and bundle:
            st.markdown("#### 🔎 Similar Movies (TF-IDF)")
            poster_grid(
                to_cards_from_tfidf_items(bundle.get("tfidf_recommendations")),
                cols=grid_cols,
                key_prefix="details_tfidf",
            )

            st.markdown("#### 🎭 More Like This (Genre)")
            poster_grid(
                bundle.get("genre_recommendations", []),
                cols=grid_cols,
                key_prefix="details_genre",
            )
        else:
            st.warning(f"Recommendations failed: {err2}")
