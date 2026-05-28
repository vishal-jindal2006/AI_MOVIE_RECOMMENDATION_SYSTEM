import streamlit as st
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity
API_KEY = "dc039555"

# Page Config
st.set_page_config(
    page_title="Netflix AI Recommender",
    page_icon="🎬",
    layout="wide"
)
# Custom Netflix Dark Theme
st.markdown(
    """
    <style>

    .stApp {
        background-color: #141414;
        color: white;
    }

    section[data-testid="stSidebar"] {
        background-color: #000000;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #E50914;
    }

    .stButton > button,
    div.stLinkButton > a {
        background-color: #E50914;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 10px 20px;
        font-weight: bold;
    }

    .stButton>button:hover {
        background-color: #b20710;
        color: white;
    }
    
    div.stLinkButton > a {
    background-color: #E50914;
    color: white;
    }   

    div[data-baseweb="select"] {
        background-color: #222222;
        color: white;
        border-radius: 8px;
    }

    p, label, span {
        color: white !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# Load Dataset (LIMITED for RAM optimization)
movies = pd.read_csv("movies.csv").head(5000)
ratings = pd.read_csv("ratings.csv").head(5000)
# User Behavior Analysis

total_users = ratings['userId'].nunique()

total_movies = movies['movieId'].nunique()

average_rating = round(
    ratings['rating'].mean(),
    2
)

most_rated_movie_id = ratings[
    'movieId'
].value_counts().idxmax()

most_rated_movie = movies[
    movies['movieId'] == most_rated_movie_id
]['title'].values[0]
# Collaborative Filtering Matrix

user_movie_matrix = ratings.pivot_table(
    index='userId',
    columns='movieId',
    values='rating'
).fillna(0)

# User Similarity

user_similarity = cosine_similarity(
    user_movie_matrix
)

# Fill missing values
movies["genres"] = movies["genres"].fillna("")

# TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words="english")

tfidf_matrix = tfidf.fit_transform(movies["genres"])

# Similarity Matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Create indices
indices = pd.Series(
    movies.index,
    index=movies["title"]
).drop_duplicates()

# Recommendation Function
def recommend(title):
   
    if title not in indices:
        return []

    idx = indices[title]

    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(
        sim_scores,
        key=lambda x: x[1],
        reverse=True
    )

    sim_scores = sim_scores[1:11]

    movie_indices = [i[0] for i in sim_scores]

    return movies.iloc[movie_indices]


def collaborative_recommend(movie_title):

    # Find movie ID
    movie_data = movies[
        movies['title'] == movie_title
    ]

    if movie_data.empty:
        return []

    movie_id = movie_data.iloc[0]['movieId']

    # Users who watched/rated movie
    users_who_liked = ratings[
        ratings['movieId'] == movie_id
    ]['userId']

    # Movies liked by similar users
    recommended_movies = ratings[
        ratings['userId'].isin(users_who_liked)
    ]['movieId'].value_counts()

    # Top recommendations
    recommended_movies = recommended_movies.head(10)

    recommended_titles = []

    for movieId in recommended_movies.index:

        title_data = movies[
            movies['movieId'] == movieId
        ]

        if not title_data.empty:

            recommended_titles.append(
                title_data.iloc[0]['title']
            )

    return recommended_titles
def fetch_poster(movie_title):

    url = (
        f"http://www.omdbapi.com/?t={movie_title}"
        f"&apikey={"dc039555"}"
    )

    data = requests.get(url).json()

    poster = data.get("Poster")

    if poster and poster != "N/A":

        return poster

    return None
# Sidebar
st.sidebar.title("🎬 Netflix AI")

st.sidebar.markdown("---")

st.sidebar.info(
    "AI Movie Recommendation System"
)

# Main Title
st.title("🎥 AI Movie Recommendation System")
# User Behavior Analysis Section

st.markdown("---")

st.header("📊 User Behavior Analysis")

col1, col2 = st.columns(2)

with col1:

    st.metric(
        "👤 Total Users",
        total_users
    )

    st.metric(
        "🎬 Total Movies",
        total_movies
    )

with col2:

    st.metric(
        "⭐ Average Rating",
        average_rating
    )

    st.metric(
        "🏆 Most Rated Movie",
        most_rated_movie
    )

st.markdown("---")

st.write(
    "Get personalized movie recommendations instantly."
)

# Movie Selection
selected_movie = st.selectbox(
    "Select Movie",
    movies["title"].values
)


# Movie Details Section

movie_info = movies[
    movies['title'] == selected_movie
].iloc[0]

st.markdown("---")

st.header("🎬 Movie Details")

st.subheader(
    movie_info['title']
)

st.info(
    f"🎭 Genres: {movie_info['genres']}"
)

# Recommendation Button
if st.button("Recommend"):

    recommendations = recommend(
        selected_movie
    )

    st.subheader(
        "🤝 Collaborative Filtering Recommendations"
    )

    collab_movies = collaborative_recommend(
        selected_movie
    )

    for movie in collab_movies:

        st.write("🎬", movie)

    st.subheader("Recommended Movies")

    cols = st.columns(2)

    for index, row in recommendations.iterrows():

        with cols[index % 2]:
            
            poster = fetch_poster(
                row['title']
            )

            if poster:

                st.image(
                    poster,
                    use_container_width=True
                )


            st.markdown(
                f"""
                ### 🎬 {row['title']}
                
                **Genres:**  
                {row['genres']}
                """
            )

            movie_name = row['title']

            youtube_search = (
                "https://www.youtube.com/results?search_query="
                + movie_name.replace(" ", "+")
                + "+official+trailer"
            )

            st.link_button(
    "▶ Watch Trailer",
    youtube_search
)
st.markdown("---")
# Trending Movies
st.subheader("🔥 Trending Movies")

popular_movies = ratings.groupby(
    "movieId"
)["rating"].mean().sort_values(
    ascending=False
).head(10)

for movie_id in popular_movies.index:

    movie_name = movies[
        movies["movieId"] == movie_id
    ]["title"].values

    if len(movie_name) > 0:
        st.write("⭐", movie_name[0])

# Footer
st.markdown("---")

st.caption(
    "Made with ❤️ using Python & Machine Learning"
)

# Contact Us Section
st.markdown("---")

st.header("📞 Contact Us")

with st.form("contact_form"):

    contact_name = st.text_input("Your Name")

    contact_email = st.text_input("Your Email")

    contact_message = st.text_area("Your Message")

    submit_button = st.form_submit_button(
        "Send Message"
    )

    if submit_button:

        with open(
            "messages.txt",
            "a",
            encoding="utf-8"
        ) as file:

            file.write(
                f"Name: {contact_name}\n"
            )

            file.write(
                f"Email: {contact_email}\n"
            )

            file.write(
                f"Message: {contact_message}\n"
            )

            file.write(
                "-" * 50 + "\n"
            )

        st.success(
            "Message sent successfully!"
        )
st.caption(
    "Made By Vishal Jindal"
)