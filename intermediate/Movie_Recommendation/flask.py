from flask import Flask
import pickle

app = Flask(__name__)

# Load ML models
movies_dict = pickle.load(open('movie_dict.pkl', 'rb'))
similarity = pickle.load(open('movies.pkl', 'rb'))

@app.route('/')
def home():
    return "ML Models Loaded Successfully!"

if __name__ == '__main__':
    app.run(debug=True)
