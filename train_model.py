# train_model.py
import pandas as pd
import joblib
from p import clean_text, create_target_dataframe, build_and_train_model

# Load dataset
df = pd.read_csv("nlp_dataset.txt")  # Replace with your actual file path
df, genre_columns = create_target_dataframe(df)

# Train and save model
model, vectorizer = build_and_train_model(df, genre_columns)
joblib.dump(model, 'model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
joblib.dump(genre_columns, 'genre_columns.pkl')

print("Model, vectorizer, and genre columns saved.")



