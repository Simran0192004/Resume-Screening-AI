from google import genai
import os

# Initialize Gemini client
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

def generate_llm_feedback(
    matched_skills,
    missing_skills,
    semantic_score,
    role="the given job role"
):
    prompt = f"""
You are an experienced technical recruiter.

Job Role:
{role}

Semantic Resume Match Score:
{semantic_score:.1f}%

Matched Skills:
{', '.join(matched_skills) if matched_skills else 'None'}

Missing Skills:
{', '.join(missing_skills) if missing_skills else 'None'}

Generate:
1. A brief resume–job match summary
2. Strengths of the candidate
3. Skill gaps to focus on
4. Practical, realistic improvement advice

Keep the tone professional, encouraging, and realistic.
Avoid buzzwords.
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text.strip()
