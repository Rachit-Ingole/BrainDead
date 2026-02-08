import streamlit as st
import pandas as pd
import pickle

@st.cache_data
def load_data():
    ratings = pd.read_csv("ratings.csv")
    movies = pd.read_csv("movies.csv")
    tags = pd.read_csv("tags.csv")

    ratings["userId"] = ratings["userId"].astype(int)
    ratings["movieId"] = ratings["movieId"].astype(int)
    movies["movieId"] = movies["movieId"].astype(int)
    tags["movieId"] = tags["movieId"].astype(int)

    movies["genre_list"] = movies["genres"].apply(
        lambda x: [] if x == "(no genres listed)" else x.split("|")
    )

    movie_tags = (
        tags.groupby("movieId")["tag"]
        .apply(lambda x: set(x.str.lower()))
        .to_dict()
    )

    return ratings, movies, movie_tags


ratings, movies, movie_tags = load_data()

@st.cache_resource
def load_model():
    with open("svd_model.pkl", "rb") as f:
        return pickle.load(f)

svd_model = load_model()


def get_user_history(user_id, min_rating=4.0, top_n=5):
    history = ratings[
        (ratings["userId"] == user_id) & (ratings["rating"] >= min_rating)
    ].merge(movies, on="movieId")

    history = history.sort_values("rating", ascending=False)
    return history[["movieId", "title", "rating", "genre_list"]].head(top_n)


def compute_match_percentage(user_id, rec_movie_id, rec_genres):
    liked_movies = ratings[
        (ratings["userId"] == user_id) & (ratings["rating"] >= 4)
    ]["movieId"].tolist()

    if not liked_movies:
        return 0.0

   
    user_genres = set()
    for mid in liked_movies:
        row = movies[movies["movieId"] == mid]
        if not row.empty:
            user_genres.update(row.iloc[0]["genre_list"])

    if not user_genres:
        return 0.0

    genre_overlap = set(rec_genres) & user_genres
    genre_score = len(genre_overlap) / len(user_genres)

   
    user_tags = set()
    for mid in liked_movies:
        user_tags.update(movie_tags.get(mid, set()))

    rec_tags = movie_tags.get(rec_movie_id, set())

    tag_bonus = 0.0
    if user_tags and rec_tags:
        overlap = user_tags & rec_tags
        tag_bonus = min(len(overlap) * 0.04, 0.2) 

    return round((genre_score + tag_bonus) * 100, 1)


def explain_recommendation(user_id, rec_movie):
    liked_movies = get_user_history(user_id)

    best_match = None
    best_overlap = []

    for _, row in liked_movies.iterrows():
        overlap = set(row["genre_list"]) & set(rec_movie["genre_list"])
        if len(overlap) > len(best_overlap):
            best_overlap = overlap
            best_match = row["title"]

    if best_match and best_overlap:
        return (
            f"Recommended because you liked **{best_match}**, "
            f"which shares genres **{', '.join(list(best_overlap)[:3])}**."
        )
    else:
        return "Recommended based on users with similar preferences."


@st.cache_data
def generate_svd_candidates(user_id):
    watched = set(
        ratings[ratings["userId"] == user_id]["movieId"].tolist()
    )

    rows = []
    for _, movie in movies.iterrows():
        mid = movie["movieId"]
        if mid in watched:
            continue

        pred = svd_model.predict(user_id, mid)
        rows.append({
            "movieId": mid,
            "title": movie["title"],
            "Predicted Rating": pred.est,
            "genre_list": movie["genre_list"]
        })

    df = pd.DataFrame(rows)
    return df.sort_values("Predicted Rating", ascending=False).head(200)


def svd_recommendations(user_id, top_n=5):
    cand_df = generate_svd_candidates(user_id)

    final_rows = []
    for _, row in cand_df.iterrows():
        match_pct = compute_match_percentage(
            user_id,
            row["movieId"],
            row["genre_list"]
        )

        norm_rating = row["Predicted Rating"] / 5.0
        final_score = 0.7 * norm_rating + 0.3 * (match_pct / 100)

        final_rows.append({
            "movieId": row["movieId"],
            "title": row["title"],
            "Predicted Rating": row["Predicted Rating"],
            "Match %": match_pct,
            "Final Score": final_score,
            "genre_list": row["genre_list"]
        })

    final_df = pd.DataFrame(final_rows)
    final_df = final_df.sort_values(
        by=["Final Score", "Predicted Rating"],
        ascending=False
    )

    return final_df.head(top_n)

st.set_page_config(page_title="ReelSense", layout="centered")

st.title("🎬 ReelSense")
st.caption("Explainable • Personalized • Hybrid Recommender")

user_id = st.number_input(
    "Enter User ID",
    min_value=int(ratings.userId.min()),
    max_value=int(ratings.userId.max()),
    value=1,
    step=1
)

top_k = st.slider("Number of recommendations", 1, 10, 5)

if st.button("Get Recommendations"):

    st.subheader("👤 User Profile Snapshot (Liked Movies)")
    history = get_user_history(user_id)

    if history.empty:
        st.info("No high-rated movies found for this user.")
    else:
        for _, row in history.iterrows():
            st.write(f"• **{row['title']}** — ⭐ {row['rating']}")

    st.subheader("🎯 Recommended Movies")
    recs = svd_recommendations(user_id, top_k)

    st.write(f"DEBUG → Returned {len(recs)} recommendations")

    for _, row in recs.iterrows():
        st.markdown(f"### 🎬 {row['title']}")
        st.write(f"⭐ **Predicted Rating:** {row['Predicted Rating']:.2f}")
        st.write(f"🎯 **Match Score:** {row['Match %']}%")
        st.progress(min(row["Match %"] / 100, 1.0))
        st.write(f"💡 {explain_recommendation(user_id, row)}")
        st.markdown("---")
