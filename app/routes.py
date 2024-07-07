
from flask import render_template, request
from app import app
import pickle

# Load the trained model
model = pickle.load(open('models/recommendation_model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    mood = request.form['mood']
    time = request.form['time']
    
    # Generate recommendations (dummy implementation)
    recommendations = ["Movie 1", "Movie 2", "Book 1"]
    
    return render_template('result.html', recommendations=recommendations)
