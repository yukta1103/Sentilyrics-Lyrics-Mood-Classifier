# Sentilyrics - Lyrics Mood Classifier

## Project Overview
Sentilyrics is a project that classifies song lyrics into different emotional categories such as fear, disgust, happiness, and more. By analyzing the textual content of lyrics, the model predicts the underlying mood, enabling applications like sentiment analysis, mood-based music recommendations, and emotional trend analysis in songs.

## How to Run the Code

### 1. Clone the repository
git clone https://github.com/yourusername/Sentilyrics-Lyrics-Mood-Classifier.git
cd Sentilyrics-Lyrics-Mood-Classifier

### 2. Set up the environment
Create and activate a Python virtual environment (optional but recommended):
python -m venv venv
venv\Scripts\activate

### 3. Install dependencies
pip install -r requirements.txt

### 4. Run Exploratory Data Analysis
jupyter notebook notebooks/exploratory.ipynb

### 5. Train the model
TF-IDF Model: 
python src/train.py --model tfidf --data data/labeled_emotions.csv

BERT Model:
python src/train.py --model bert --data data/labeled_emotions.csv

### 6. Run for predictions
python src/inference.py --model_type tfidf --model_path model.pkl --vectorizer_path vectorizer.pkl --input_file src/input_texts.txt

### 7. Launch the streamlit web app
streamlit run app/app.py





