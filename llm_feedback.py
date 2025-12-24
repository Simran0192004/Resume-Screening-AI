import os
import google.generativeai as genai

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing. Check Streamlit Secrets.")

genai.configure(api_key=API_KEY)

def generate_llm_feedback(resume_text, job_description):
    model = genai.GenerativeModel("gemini-pro")

    prompt = f"""
You are a professional technical recruiter.

Analyze the resume against the job description and provide:
1. Key strengths
2. Skill gaps
3. Clear improvement suggestions

Resume:
{resume_text}

Job Description:
{job_description}
"""

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ LLM Error: {str(e)}"
