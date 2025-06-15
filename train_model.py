import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pickle


df = pd.read_csv(r"C:\Users\shaik\archive spotify\dataset.csv")  


features = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness']
X = df[features]


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)


pickle.dump((scaler, kmeans), open("genre_model.pkl", "wb"))

print("✅ Model trained and saved successfully!")
