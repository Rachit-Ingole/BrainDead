from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import pickle
import numpy as np
from numpy.linalg import norm
from collections import Counter

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)



ratings = pd.read_csv("ratings.csv")
movies = pd.read_csv("movies.csv")
tags = pd.read_csv("tags.csv")
links = pd.read_csv("links.csv")  


tags['tag'] = tags['tag'].astype(str).str.lower()
movie_tags_map = tags.groupby('movieId')['tag'].apply(list).to_dict()


movies = movies.merge(links, on="movieId", how="left")


movie_stats = ratings.groupby('movieId').agg({'rating': ['count', 'mean']})
movie_stats.columns = ['count', 'mean']
C = movie_stats['mean'].mean()
m = movie_stats['count'].quantile(0.70)

def weighted_rating(x):
    v = x['count']
    R = x['mean']
    return (v/(v+m) * R) + (m/(m+v) * C)

movie_stats['weighted_score'] = movie_stats.apply(weighted_rating, axis=1)
weighted_scores = movie_stats['weighted_score'].to_dict()


movies["genre_list"] = movies["genres"].apply(
    lambda x: [] if x == "(no genres listed)" else x.split("|")
)


with open("svd_model.pkl", "rb") as f:
    svd_model = pickle.load(f)


def get_user_tag_profile(user_id):
    """Analyzes the user's history to find their favorite TAGS."""
    user_history = ratings[(ratings["userId"] == user_id) & (ratings["rating"] >= 4.0)]
    liked_movie_ids = user_history["movieId"].tolist()
    
    all_user_tags = []
    for mid in liked_movie_ids:
        if mid in movie_tags_map:
            all_user_tags.extend(movie_tags_map[mid])
            
   
    if not all_user_tags: return set()
    return set([t for t, c in Counter(all_user_tags).most_common(15)])

def recommend(user_id: int, top_k: int):

    user_ratings = ratings[ratings["userId"] == user_id]
    watched_ids = set(user_ratings["movieId"])
    user_fav_tags = get_user_tag_profile(user_id) 
    

    candidate_movies = movies[~movies["movieId"].isin(watched_ids)]
    
    predictions = []

  
    for row in candidate_movies.itertuples(index=False):
        try:
            svd_pred = svd_model.predict(user_id, row.movieId).est
            
            
            global_score = weighted_scores.get(row.movieId, 3.0)
            
            hybrid_rating = (svd_pred * 0.7) + (global_score * 0.3)
            
            display_rating = min(5.0, hybrid_rating)
            
            if display_rating < 3.0: continue

            current_tags = set(movie_tags_map.get(row.movieId, []))
            overlap = user_fav_tags.intersection(current_tags)
            
            tag_bonus = len(overlap) * 0.25
            sort_score = display_rating + tag_bonus

            genre = row.genre_list[0] if row.genre_list else "Film"
            
            if len(overlap) > 0:
                top_match = list(overlap)[0]
                explanation = f"Recommended because you like '{top_match}' movies."
                base_match = 75 + (len(overlap) * 5) 
                variance = (svd_pred % 1) * 10 
                match_pct = min(99.5, base_match + variance)

            elif svd_pred > 4.5:
                explanation = f"Highly predicted match based on your viewing patterns."
                match_pct = min(95, (svd_pred/5)*100)
            else:
                explanation = f"Critically acclaimed {genre} you might have missed."
                match_pct = 60 + (global_score * 5) + ((svd_pred % 1) * 5)

            tmdb_id = int(row.tmdbId) if pd.notna(row.tmdbId) else None

            predictions.append({
                "movieId": row.movieId,
                "title": row.title,
                "tmdbId": tmdb_id, #
                "predicted_rating": round(display_rating, 2), 
                "match_percentage": round(match_pct, 1),
                "explanation": explanation,
                "sort_score": sort_score 
            })
        except:
            continue


    predictions.sort(key=lambda x: x["sort_score"], reverse=True)
    
    for p in predictions:
        del p["sort_score"]
    
    return predictions[:top_k]

@app.get("/")
def root():
    return {"message": "ReelSense Hybrid (Pattern Learning) API Running"}

@app.get("/recommendations")
def get_recommendations(user_id: int, top_k: int = 10):
    return recommend(user_id, top_k)