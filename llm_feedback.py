import os
import google.generativeai as genai


def generate_llm_feedback(resume_text, job_description):
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return "Gemini API key not found. Please configure it in Streamlit secrets."

    genai.configure(api_key=api_key)

    model = genai.GenerativeModel("gemini-pro")

    prompt = f"""
You are a professional technical recruiter.

Analyze the resume against the job description and provide:
1. Strengths
2. Skill gaps
3. Improvement suggestions

Resume:
{resume_text}

Job Description:
{job_description}
"""

    response = model.generate_content(prompt)
    return response.text
