import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load models and data
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
cosine_sim = joblib.load('cosine_similarity.pkl')
df = pd.read_csv('job_descriptions_cleaned.csv')

# Create the TF-IDF matrix from the job descriptions in the DataFrame
tfidf_matrix = tfidf_vectorizer.transform(df['Job Description'])

def recommend_jobs(job_title):
    if job_title not in df['Job Title'].values:
        return "Job title not found."
    idx = df[df['Job Title'] == job_title].index[0]
    job_description = df.loc[idx, 'Job Description']
    job_desc_vector = tfidf_vectorizer.transform([job_description])
    sim_scores = cosine_similarity(job_desc_vector, tfidf_matrix).flatten()
    sim_scores_idx = sim_scores.argsort()[-10:][::-1]
    similar_jobs = df.iloc[sim_scores_idx]
    return similar_jobs[['Job Title', 'Company']]

# Streamlit UI
st.title('Job Recommendation System')

job_title = st.text_input('Enter Job Title')

if st.button('Recommend Jobs'):
    recommended_jobs = recommend_jobs(job_title)
    if isinstance(recommended_jobs, str):
        st.write(recommended_jobs)
    else:
        st.write(recommended_jobs)
