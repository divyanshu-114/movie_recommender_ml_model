import streamlit as st
import pickle
import pandas as pd

# Load model
movies = pickle.load(open('model/movies.pkl','rb'))
similarity = pickle.load(open('model/similarity.pkl','rb'))

# Recommendation function
def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = similarity[index]
    
    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]
    
    recommended = []
    for i in movie_list:
        recommended.append(movies.iloc[i[0]].title)
    
    return recommended

# ===== STREAMLIT UI =====

st.title("🎬 Movie Recommendation System")

selected_movie = st.selectbox(
    "Select a movie",
    movies['title'].values
)

if st.button("Recommend"):
    recommendations = recommend(selected_movie)
    
    st.subheader("Recommended Movies:")
    for movie in recommendations:
        st.write(movie)