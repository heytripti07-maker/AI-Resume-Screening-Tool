import streamlit as st
import fitz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("Smart Resume Screening Tool")

resume = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
jd = st.text_area("Paste Job Description")

if st.button("Check Match"):
    if resume and jd:

        # Resume PDF ka text read karo
        pdf = fitz.open(stream=resume.read(), filetype="pdf")
        resume_text = ""

        for page in pdf:
            resume_text += page.get_text()

        # Resume aur JD compare karo
        text = [resume_text, jd]

        tfidf = TfidfVectorizer()
        vectors = tfidf.fit_transform(text)

        score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]

        st.success("Resume Uploaded Successfully!")
        st.write("Match Score:", round(score * 100, 2), "%")

    else:
        st.warning("Please upload resume and enter Job Description.")