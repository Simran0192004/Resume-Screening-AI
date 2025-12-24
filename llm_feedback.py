import os
from google import genai

def generate_llm_feedback(resume_text, job_description):

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "⚠️ GEMINI_API_KEY not found in environment variables."

    client = genai.Client(api_key=api_key)

    prompt = f"""
You are a professional technical recruiter.

Analyze the resume against the job description and provide:

1. Key strengths
2. Skill gaps
3. Improvement suggestions

Resume:
{resume_text}

Job Description:
{job_description}
"""

    response = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=prompt
    )

    return response.text
