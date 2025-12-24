import os
import google.generativeai as genai

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def generate_llm_feedback(resume_text, job_description):
    model = genai.GenerativeModel("gemini-3-pro-preview")

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
