import os
import google.generativeai as genai

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def generate_llm_feedback(resume_text, job_description):
    model = genai.GenerativeModel("gemini-1.5-flash")

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

Keep the tone concise, practical, and realistic.
"""

    response = model.generate_content(prompt)
    return response.text
