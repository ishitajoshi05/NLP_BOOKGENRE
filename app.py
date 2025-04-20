from flask import Flask, request, jsonify, send_file
import joblib
from p import clean_text, combine_minor_genres

app = Flask(__name__)
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
genre_columns = joblib.load("genre_columns.pkl")

@app.route('/')
def home():
    return send_file('a.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    synopsis = data.get("synopsis", "")
    if not synopsis:
        return jsonify({"error": "No synopsis provided"}), 400

    cleaned = clean_text(synopsis)
    X = vectorizer.transform([cleaned])
    y_pred = model.predict(X)[0]

    total = y_pred.sum()
    y_pred_pct = (y_pred / total * 100) if total > 0 else y_pred
    prediction = {genre: round(float(val), 2) for genre, val in zip(genre_columns, y_pred_pct)}
    combined = combine_minor_genres(prediction)

    return jsonify({
        "genres": combined,
        "subgenres": {}
    })

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=7860)
