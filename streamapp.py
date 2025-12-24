import streamlit as st
import nltk
import re

from sentence_transformers import SentenceTransformer, util
from llm_feedback import generate_llm_feedback

nltk.download('stopwords')

# -------------------------------------
# Page Config
# -------------------------------------
st.set_page_config(
    page_title="AI Resume Screening",
    page_icon="📄",
    layout="wide"
)

st.title("📄 AI-Powered Resume Screening System")
st.caption("Semantic similarity • Skill gap analysis • LLM recruiter feedback")

st.divider()

# -------------------------------------
# Text Preprocessing
# -------------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    return text

 def extract_role(job_text):
                lines = job_text.split('\n')
                for line in lines[:10]:  # only scan top section
                    if any(keyword in line.lower() for keyword in [
                        'engineer', 'developer', 'scientist', 'analyst',
                        'intern', 'manager', 'researcher'
                    ]):
                        return line.strip()
                return "the given role"

# -------------------------------------
# Skill Extraction
# -------------------------------------
skill_keywords = [
    'python','java','c++','sql','pandas','numpy','tensorflow','pytorch',
    'machine learning','deep learning','nlp','computer vision','data analysis',
    'docker','kubernetes','aws','git','linux','excel','tableau','matplotlib'
]

def extract_skills(text, skills_list):
    text = text.lower()
    return [skill for skill in skills_list if skill in text]

# -------------------------------------
# Input Section
# -------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Upload Resume")
    resume_file = st.file_uploader(
        "Accepted formats: .txt",
        type=["txt"]
    )

with col2:
    st.subheader("Job Description")
    job_text = st.text_area(
        "Paste the job description here",
        height=250
    )

analyze = st.button("🔍 Analyze Resume")

# -------------------------------------
# Session State Init
# -------------------------------------
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False

# -------------------------------------
# Analysis Logic
# -------------------------------------
if analyze:
    role = extract_role(job_text)
    if resume_file is None or job_text.strip() == "":
        st.warning("Please upload a resume and paste a job description.")
    else:
        with st.spinner("Analyzing resume..."):

            # Read resume text
            resume_text = resume_file.read().decode("utf-8")

            # Clean text
            resume_clean = clean_text(resume_text)
            job_clean = clean_text(job_text)

            # Semantic similarity
            model = SentenceTransformer('all-MiniLM-L6-v2')
            embeddings = model.encode(
                [resume_text, job_text],
                convert_to_tensor=True
            )

            similarity_score = util.pytorch_cos_sim(
                embeddings[0],
                embeddings[1]
            ).item() * 100

            # Skill extraction
            resume_skills = extract_skills(resume_clean, skill_keywords)
            jd_skills = extract_skills(job_clean, skill_keywords)

            matched_skills = list(set(resume_skills) & set(jd_skills))
            missing_skills = list(set(jd_skills) - set(resume_skills))

            # Skill match percentage
            skill_match_percent = (
                len(matched_skills) / len(jd_skills) * 100
            ) if jd_skills else 0

            # -------------------------------------
            # LLM Context Construction
            # -------------------------------------
            llm_resume_context = f"""
            Role Applied For: {role}

            Semantic Similarity Score: {similarity_score:.2f}%

            Skill Match Percentage: {skill_match_percent:.2f}%    

            Matched Skills:
            {', '.join(matched_skills) if matched_skills else 'None'}

            Missing Skills:
            {', '.join(missing_skills) if missing_skills else 'None'}
            """

            # LLM feedback
            llm_feedback = generate_llm_feedback(
               resume_text=llm_resume_context,
               job_description=job_text
            )

            # Store in session state
            st.session_state.analysis_done = True
            st.session_state.semantic_score = similarity_score
            st.session_state.matched_skills = matched_skills
            st.session_state.missing_skills = missing_skills
            st.session_state.skill_match_percent = skill_match_percent
            st.session_state.llm_feedback = llm_feedback

# -------------------------------------
# Results Section
# -------------------------------------
if st.session_state.analysis_done:

    st.divider()
    st.header("📊 Screening Results")

    st.subheader("🎯 Target Role")
    st.info(role)

    # Match Score
    st.subheader("Resume–Job Match Score")
    st.metric(
        "Semantic Similarity",
        f"{st.session_state.semantic_score:.2f}%"
    )
    st.progress(st.session_state.semantic_score / 100)

    # Skills
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("✅ Matched Skills")
        if st.session_state.matched_skills:
            for skill in st.session_state.matched_skills:
                st.write(f"• {skill}")
        else:
            st.write("No matched skills found.")

    with col2:
        st.subheader("❌ Missing Skills")
        if st.session_state.missing_skills:
            for skill in st.session_state.missing_skills:
                st.write(f"• {skill}")
        else:
            st.write("No major skill gaps detected.")

    # LLM Feedback
    st.subheader("🤖 AI Recruiter Feedback")
    st.write(st.session_state.llm_feedback)

    # Resume preview
    with st.expander("📄 View Resume Text"):
        st.write(resume_text)
