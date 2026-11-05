import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
def load_data():
    df = pd.read_csv('dataset.csv')
    df['features'] = (
        df['Gift_Name'].astype(str) + ' '
        +df['Occasion'].astype(str) + ' '
        + df['Recipient_Age_Group'].astype(str) + ' '
        + df['Category'].astype(str)
    )
    return df
    
def get_recommendations(user_input, dataframe, top_n=3):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(dataframe['features'])
    user_input_vector = vectorizer.transform([user_input])
    similarity_scores = cosine_similarity(user_input_vector, tfidf_matrix)
    top_indices = similarity_scores.argsort()[0][-top_n:][::-1]
    recommendations = dataframe.iloc[top_indices]
    return recommendations

st.title("Gift Recommendation System")
user_input = st.text_input("Enter your gift preferences (e.g., 'Birthday gift for a 10-year-old boy'):")
if st.button("Get Recommendations"):
    if user_input:
        df = load_data()
        recommendations = get_recommendations(user_input, df)
        st.write("Top Gift Recommendations:")
        for index, row in recommendations.iterrows():
            st.write(f"Gift Name: {row['Gift_Name']}, Occasion: {row['Occasion']}, Age Group: {row['Recipient_Age_Group']}, Category: {row['Category']}")
    else:
        st.write("Please enter your gift preferences to get recommendations.")