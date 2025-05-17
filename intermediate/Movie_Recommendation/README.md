# 🎬 Cineeseek – Movie Recommendation System

**Cineeseek** (also known as **CineMatch**) is a Machine Learning-powered, content-based movie recommendation web application. Just enter the name of a movie you love, and Cineeseek will suggest similar movies based on metadata like genre, cast, keywords, and crew – complete with poster previews using the TMDB API.

---

## 🔗 Live Demo

👉 [https://cineeseek.vercel.app/](https://cineeseek.vercel.app/) – **Please do visit and try it out!**

---

## 🚀 Features

- 🎭 **Intuitive UI** – Clean, modern interface with animated glowing movie titles and card layout
- 🖼️ **Poster Previews** – Movie posters fetched dynamically using the TMDB API
- 🧠 **ML-Powered Recommendations** – Content-based filtering using cosine similarity on movie feature vectors
- ⚡ **Fast Response** – Instant results using a precomputed similarity matrix
- 🎥 **100+ Movie Dataset** – Recommendations are based on rich movie metadata including genre, cast, crew, and keywords
- ✨ **Styled Interface** – Custom CSS, glowing effects, and themed visuals enhance the user experience
- 🌐 **Deployed on Vercel** – Fully functional and live on the web

---

## 🛠️ Tech Stack

### 🔧 Frontend:
- Streamlit (Python-based web interface)
- Custom HTML/CSS for visual enhancements

### 🔙 Backend:
- Python
- pandas, pickle
- scikit-learn (CountVectorizer, cosine_similarity)

### 🔗 External API:
- [TMDB API](https://www.themoviedb.org/documentation/api) – used for fetching poster images

---

## 🧠 Machine Learning Overview

### ✅ Technique:
- **Content-Based Filtering** using **Cosine Similarity**

### 📦 Data Preprocessing:
- Merged and cleaned fields like *genres*, *keywords*, *cast*, *crew* into a single `tags` column
- Applied stemming to normalize words
- Vectorized the `tags` column using `CountVectorizer`

### 🔢 Vectorization and Similarity:
- Each movie is converted into a numeric feature vector
- Cosine similarity is computed between these vectors
- Top 5 most similar movies are returned as recommendations

### 💾 Output Files:
- `similarity.pkl` – precomputed cosine similarity matrix
- `movie_dict.pkl` – contains movie metadata for fast lookup

---

## 🖼️ TMDB Poster API Usage

Posters are dynamically fetched using:

https://image.tmdb.org/t/p/w500/<poster_path>

Get your API key by signing up at [TMDB](https://www.themoviedb.org/signup).

---

## 🧪 Run Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/cineeseek.git
   cd cineeseek

Install dependencies:
pip install -r requirements.txt
Add your TMDB API key in the appropriate file (as an environment variable or directly in code)

Run the app:
streamlit run app.py

🙌 Acknowledgements
The Movie Database (TMDB)

scikit-learn, pandas, and the Streamlit community

Inspiration from open-source recommendation systems

📫 Contact
D. Ravi Kiran

🌐 Live App: https://cineeseek.vercel.app/

💼 LinkedIn: your-linkedin

🐦 Twitter: @yourhandle

📃 License
Licensed under the MIT License – free to use, modify, and distribute.


---

Let me know if you'd like help creating a banner/screenshot image or deploying this `README.md` to GitHub!
