🎬 CineMatch – A Movie Recommendation System
CineMatch is a content-based movie recommendation web application developed using Streamlit and powered by a machine learning model that analyzes movie metadata to suggest similar films.

🚀 Features:
🎭 Intuitive UI: Clean, modern interface with animated glowing movie titles and card layout.

🧠 ML-Powered Recommendations: Uses cosine similarity on feature-engineered vectors to suggest the top 5 similar movies.

🎥 100+ Movie Dataset: Works with a dataset containing movie titles, genres, cast, crew, and keywords.

⚡ Fast Response: Uses precomputed similarity matrix for instant recommendations.

✨ Fully Styled: Custom CSS for themed visuals and branding.

🧠 Machine Learning Overview:
✅ Technique: Content-Based Filtering using Cosine Similarity.

📦 Data Preprocessing:

Text data like genres, keywords, cast, crew are cleaned and combined into a single tags column.

Applied Stemming and Vectorization (using CountVectorizer).

🔢 Vectorization:

Each movie is converted into a numeric feature vector.

Cosine similarity is computed between these vectors to measure content closeness.

💾 Preprocessing Output:

A similarity.pkl file (cosine similarity matrix).

A movie_dict.pkl file (containing movie metadata).

📌 Tech Stack:
Frontend: Streamlit (Python)

Backend: Python (pandas, pickle, sklearn)

ML Libraries: scikit-learn (CountVectorizer, cosine_similarity)

Styling: Custom HTML/CSS for UI enhancements
