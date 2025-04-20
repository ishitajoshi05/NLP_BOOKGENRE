import pandas as pd
import ast
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def clean_text(text):
    """
    Lowercase conversion and removal of non-alphabetic characters.
    """
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def convert_genre_ratios(ratio_str):
    """
    Convert string representation of a dictionary into an actual dictionary.
    """
    try:
        return ast.literal_eval(ratio_str)
    except Exception:
        return {}

def create_target_dataframe(df, target_column='Genre_Ratios'):
    """
    Convert the Genre_Ratios column into individual target columns for each genre.
    Missing genre values are filled with 0.
    """
    df['Genre_Ratios_dict'] = df[target_column].apply(convert_genre_ratios)
    all_genres = set()
    for d in df['Genre_Ratios_dict']:
        all_genres.update(d.keys())
    all_genres = sorted(list(all_genres))
    for genre in all_genres:
        df[genre] = df['Genre_Ratios_dict'].apply(lambda d: d.get(genre, 0))
    return df, all_genres

def build_and_train_model(df, genre_columns, text_column='Synopsis'):
    """
    Build TF-IDF features from text and train a multi-output regression model.
    """
    df['Cleaned_Text'] = df[text_column].apply(clean_text)

    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['Cleaned_Text'])

    y = df[genre_columns].values.astype(float)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    base_model = RandomForestRegressor(n_estimators=100, random_state=42)
    model = MultiOutputRegressor(base_model)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print("Test Mean Squared Error:", mse)

    return model, vectorizer

def predict_genre_ratios(synopsis, model, vectorizer, genre_columns):
    """
    Given a new book synopsis, preprocess it, extract TF-IDF features,
    and predict the genre ratios.
    """
    cleaned = clean_text(synopsis)
    X_new = vectorizer.transform([cleaned])
    y_pred = model.predict(X_new)[0]

    total = y_pred.sum()
    if total > 0:
        y_pred_pct = (y_pred / total) * 100
    else:
        y_pred_pct = y_pred

    prediction = {genre: round(float(ratio), 2) for genre, ratio in zip(genre_columns, y_pred_pct)}
    return prediction

def combine_minor_genres(prediction, top_n=3):
    """
    Combine minor genres into a single "Other" category.
    Only the top_n genres (based on predicted percentage) are kept,
    the rest are summed up as "Other".
    """
    filtered = {genre: ratio for genre, ratio in prediction.items() if ratio > 0}
    if len(filtered) <= top_n:
        return filtered

    sorted_genres = sorted(filtered.items(), key=lambda x: x[1], reverse=True)
    major_genres = dict(sorted_genres[:top_n])
    other_sum = sum([ratio for _, ratio in sorted_genres[top_n:]])
    if other_sum > 0:
        major_genres["Other"] = round(other_sum, 2)
    return major_genres

