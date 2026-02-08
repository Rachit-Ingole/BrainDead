import pandas as pd
import pickle
from surprise import Dataset, Reader, SVD

print("Loading datasets...")
ratings = pd.read_csv("ratings.csv")

reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings[["userId", "movieId", "rating"]], reader)

print("Training SVD Model (Calculating Vectors)...")

trainset = data.build_full_trainset()

model = SVD(n_factors=100, random_state=42) 
model.fit(trainset)

print("Saving model to 'svd_model.pkl'...")
with open("svd_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Success! You can now restart your API.")