import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# Load dataset
df = pd.read_csv('data/dataset.csv')

# Preprocess data
# [Include preprocessing steps here]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(df['feature'], df['label'], test_size=0.2, random_state=42)

# Vectorize text
vectorizer = TfidfVectorizer()
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train_vect, y_train)

# Save the model
pickle.dump(model, open('models/recommendation_model.pkl', 'wb'))
