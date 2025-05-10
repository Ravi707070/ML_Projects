import streamlit as st
import pickle
import pandas as pd

# Set page configuration
st.set_page_config(
    page_title="CineMatch",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items=None
)

# Load data with caching
@st.cache_data
def load_data():
    try:
        movies_dict = pickle.load(open('movie_dict.pkl', 'rb'))
        movies = pd.DataFrame(movies_dict)
        similarity = pickle.load(open('similarity.pkl', 'rb'))
        return movies, similarity
    except Exception as e:
        # Create dummy data if files not found (for demonstration)
        dummy_data = {
            'movie_id': list(range(1, 101)),
            'title': [f"Example Movie {i}" for i in range(1, 101)],
            'tags': ["Action, Adventure" for _ in range(100)]
        }
        dummy_movies = pd.DataFrame(dummy_data)
        dummy_similarity = [[0.0] * 100 for _ in range(100)]
        return dummy_movies, dummy_similarity

# Generate mock ratings and tags for demo purposes
def get_mock_rating():
    import random
    return round(random.uniform(7.5, 9.5), 1)

def get_mock_tags(movie_title):
    import random
    all_tags = ["Action", "Adventure", "Comedy", "Drama", "Sci-Fi", "Thriller", 
               "Romance", "Fantasy", "Horror", "Mystery", "Animation", "Crime"]
    # Use movie title to create deterministic but seemingly random tags
    seed = sum(ord(c) for c in movie_title)
    random.seed(seed)
    num_tags = random.randint(2, 4)
    return random.sample(all_tags, num_tags)

def get_mock_match_percentage(index):
    # Higher match percentage for higher ranked recommendations
    base = 98 - (index * 3)
    import random
    return base + random.randint(-2, 2)

# Recommendation function
def recommend(movie, movies, similarity):
    try:
        movie_index = movies[movies['title'] == movie].index[0]
        distances = similarity[movie_index]
        movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

        recommended_movies = []
        for i in movies_list:
            recommended_movies.append({
                'title': movies.iloc[i[0]].title,
                'similarity': i[1]
            })
        
        return recommended_movies
    except Exception as e:
        st.error(f"Error in recommendation: {e}")
        return []

# Display recommendations using Streamlit's native components
def display_recommendations(recommendations, show_tags):
    if recommendations:
        # Display recommendations in a grid layout
        st.markdown("### Recommendations")
        cols = st.columns(3)  # Display 3 recommendations per row

        for i, movie in enumerate(recommendations):
            with cols[i % 3]:  # Cycle through columns
                title = movie['title']
                mock_rating = get_mock_rating()
                match_percentage = get_mock_match_percentage(i)
                tags = get_mock_tags(title)

                st.markdown(f"**{i+1}. {title}**")
                if show_tags:
                    st.markdown("Genres: " + ", ".join(tags))
                st.markdown(f"Rating: {mock_rating}/10")
                st.markdown(f"Match: {match_percentage}%")

                # Add action buttons
                st.button(f"Watch '{title}'", key=f"watch_{i}")
                st.button("Save to Watchlist", key=f"save_{i}")
    else:
        st.markdown("No recommendations found. Try another movie.")

# Main function
def main():
    # Load data
    movies, similarity = load_data()
    
    # App header
    st.title("🎬 CineMatch")
    st.subheader("Find your next favorite movie")
    
    # Movie selection
    selected_movie_name = st.selectbox(
        "Choose a movie you enjoyed",
        options=movies['title'].values
    )
    
    # Number of recommendations and tag visibility
    col1, col2 = st.columns([1, 1])
    with col1:
        num_recommendations = st.slider("Number of recommendations:", min_value=3, max_value=8, value=5)
    with col2:
        show_tags = st.checkbox("Show genres", value=True)
    
    # Recommendation button
    if st.button("Find My Perfect Matches"):
        with st.spinner("Finding your perfect movie matches..."):
            recommendations = recommend(selected_movie_name, movies, similarity)[:num_recommendations]
            display_recommendations(recommendations, show_tags)

# Run the app
if __name__ == "__main__":
    main()