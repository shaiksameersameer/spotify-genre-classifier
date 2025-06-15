from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import os

app = Flask(__name__)


scaler, model = pickle.load(open("genre_model.pkl", "rb"))


genre_mapping = {
    0: "Pop",
    1: "Rock",
    2: "Jazz",
    3: "Classical",
    4: "Hip-Hop"
}


songs_dir = os.path.join("static", "songs")
songs = {
    "Pop": os.path.join(songs_dir, "pop_song.mp3"),
    "Rock": os.path.join(songs_dir, "rock_song.mp3"),
    "Jazz": os.path.join(songs_dir, "jazz_song.mp3"),
    "Classical": os.path.join(songs_dir, "classical_song.mp3"),
    "Hip-Hop": os.path.join(songs_dir, "hiphop_song.mp3")
}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    input_features = np.array([data["features"]]).reshape(1, -1)
    input_scaled = scaler.transform(input_features)
    prediction = model.predict(input_scaled)[0]

    predicted_genre = genre_mapping.get(int(prediction), "Unknown")
    song_path = f"/static/songs/{os.path.basename(songs.get(predicted_genre, ''))}"

    return jsonify({"genre_cluster": int(prediction), "genre": predicted_genre, "song": song_path})

if __name__ == "__main__":
    app.run(debug=True)
