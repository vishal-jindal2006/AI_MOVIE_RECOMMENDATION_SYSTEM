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
    
    header {
    visibility: hidden;
    }
    
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
        max-width: 95%;
    }

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
@st.cache_data
def load_movies():

    file_id = "1FVzBtAVnyLrKvq3wMSfAIkAiSaiRBEWI"

    url = f"https://drive.google.com/uc?id={file_id}"

    output = "tmdb_temp.csv"

    gdown.download(url, output, quiet=False)

    return pd.read_csv(output).head(9000)

movies = load_movies()

movies = movies[
    [
        'title',
        'genres',
        'overview',
        'poster_path',
        'vote_average',
        'release_date'
    ]
]

ratings = pd.read_csv("ratings.csv").head(5000)
# User Behavior Analysis

total_users = ratings['userId'].nunique()

total_movies = movies['title'].nunique()

average_rating = round(
    ratings['rating'].mean(),
    2
)

most_rated_movie = "Popular TMDB Movie"
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

    sim_scores = list(enumerate(cosine_sim[idx].flatten()))

    sim_scores = sorted(
        sim_scores,
        key=lambda x: x[1],
        reverse=True
    )

    sim_scores = sim_scores[1:8]

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

st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/0/08/Netflix_2015_logo.svg",
    width=140
)

st.sidebar.title("🎬 Netflix AI")
menu = st.sidebar.radio(
    "📌 Navigation",
    [
        "🏠 Home",
        "🔥 Trending",
        "⭐ Top Rated",
        "ℹ️ About"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
    "AI Movie Recommendation System"
)

# Main Title
if menu == "🏠 Home":
    
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
    # Search Movie

    search_movie = st.text_input(
        "🔍 Search Movie"
    )

    # Genre Filter

    genre_list = [
        "All",
        "Action",
        "Comedy",
        "Drama",
        "Horror",
        "Romance",
        "Thriller",
        "Adventure",
        "Animation",
        "Sci-Fi"
    ]

    selected_genre = st.selectbox(
        "🎭 Select Genre",
        genre_list
    )

    # Filter Movies

    filtered_movies = movies.copy()

    if selected_genre != "All":

        filtered_movies = filtered_movies[
            filtered_movies['genres']
            .str.contains(
                selected_genre,
                case=False,
                na=False
            )
        ]

    if search_movie:

        filtered_movies = filtered_movies[
            filtered_movies['title']
            .str.contains(
                search_movie,
                case=False,
                na=False
            )
        ]

    movie_list = filtered_movies['title'].values.tolist()

    movie_list.insert(0, "Select a Movie")

    selected_movie = st.selectbox(
        "🎬 Select Movie",
        movie_list
    )

    # Movie Details Section

    if selected_movie != "Select a Movie":

        movie_info = movies[
            movies['title'] == selected_movie
        ].iloc[0]

        st.subheader("🎬 Movie Details")

        st.write(movie_info['title'])


    # Recommendation Button
    if st.button("Recommend") and selected_movie != "Select a Movie":
        recommendations = recommend(
            selected_movie
        )

        st.subheader("Recommended Movies")

        cols = st.columns(2)

        for index, row in recommendations.iterrows():

            with cols[index % 2]:
                
                if pd.notna(row['poster_path']):

                    poster_url = (
                        "https://image.tmdb.org/t/p/w500"
                        + str(row['poster_path'])
                    )

                    st.image(
                        poster_url,
                        width=180
                    )

                else:

                    st.write("Poster Not Available")

                st.markdown(
                        f"""
                        ### 🎬 {row['title']}
                        
                        **Genres:**  
                        {row['genres']}

                        ⭐ Rating: {row['vote_average']}

                        📅 Release Date: {row['release_date']}

                        📝 Overview:  
                        {row['overview'][:200]}...
                        """
                )

                youtube_search = (
                    "https://www.youtube.com/results?search_query="
                    + row['title'].replace(" ", "+")
                    + "+official+trailer"
    )

                st.link_button(
        "▶ Watch Trailer",
        youtube_search
    )
        st.markdown("---")
        
    if menu == "🏠 Home":
        st.header("📞 Contact Us")
        with st.form("contact_form_home"):

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
    st.markdown("---")
    # Trending Movies
if menu == "🔥 Trending":

    st.header("🔥 Trending Movies")

    trending_movies = movies.sample(5)

    cols = st.columns(5)

    for idx, (_, movie) in enumerate(trending_movies.iterrows()):

        with cols[idx]:

            if pd.notna(movie["poster_path"]):

                poster_url = (
                    "https://image.tmdb.org/t/p/w500"
                    + str(movie["poster_path"])
                )

                st.image(
                    poster_url,
                    use_container_width=True
                )

                st.write(
                    f"⭐ {movie['title']}"
                )
if menu == "⭐ Top Rated":

    st.header("⭐ Top Rated Movies")

    top_movies = movies.sort_values(
        by="vote_average",
        ascending=False
    ).head(10)

    for _, row in top_movies.iterrows():

        st.markdown(
            f"""
            <div style="
                background: linear-gradient(
                    90deg,
                    #0f3d2e,
                    #071a12
                );
                padding: 15px;
                border-radius: 10px;
                margin-bottom: 10px;
                color: white;
                font-size: 18px;
            ">
            🎬 {row['title']}
            <span style="float:right;">
            ⭐ {round(row['vote_average'],1)}
            </span>
            </div>
            """,
            unsafe_allow_html=True
        )
if menu == "ℹ️ About":

    st.title("ℹ️ About This Project")

    st.markdown(
        """
        ## 🎬 AI Movie Recommendation System

        This project is a Netflix-inspired Movie Recommendation System
        developed using Python, Machine Learning, and Streamlit.

        ### 🚀 Features
        - Movie Recommendations
        - Genre Filtering
        - Search Functionality
        - Trending Movies
        - Top Rated Movies
        - Real Movie Posters
        - YouTube Trailer Support
        - Netflix Style UI

        ### 🧠 Technologies Used
        - Python
        - Pandas
        - Scikit-learn
        - Streamlit
        - TMDB Dataset
        - Machine Learning

        ### 🤖 Recommendation Technique
        This system uses Content-Based Filtering to recommend
        similar movies based on genres and movie features.

        ### 👨‍💻 Developed By
        Vishal Jindal
        """
    )

    st.image(
        "https://images.unsplash.com/photo-1489599849927-2ee91cede3ba",
        use_container_width=True
    )