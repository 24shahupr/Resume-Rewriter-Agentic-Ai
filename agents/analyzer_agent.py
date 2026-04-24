import json
from utils.groq_client import GroqClient

ANALYZER_SYSTEM_PROMPT = """You are an expert Resume Analyzer. 
Your task is to parse a raw resume text and extract key information into a structured JSON format.
Be extremely thorough. Do not skip any sections, names, contact details, jobs, or projects.

STRICT EXTRACTION RULES:
1. PERSONAL INFO: Look for Name, Email, Phone, Location, and LinkedIn. If found, extract them. If not found, use empty strings.
2. EXPERIENCE & PROJECTS: Extract ALL work experience AND significant projects. 
   - For student resumes, treat "Projects" with the same level of detail as work experience, putting them in the "experience" list.
   - Do NOT merge projects. If there are 3 projects, there should be 3 entries in the "experience" list.
3. SKILLS: Extract every single skill mentioned.
4. EDUCATION: Extract degree, institution, and dates.
5. CERTIFICATIONS & ACHIEVEMENTS: Extract them exactly as written.
6. MISC: Extract "Strengths", "Languages Known", and "Personal Details" (like DOB) into the "personal_details" and "misc" fields.

Return EXACTLY this JSON:
{
  "personal_info": {
    "name": "Full Name",
    "location": "City, Country",
    "email": "email@example.com",
    "phone": "+1234567890",
    "linkedin": "linkedin.com/in/username",
    "dob": "Date of Birth if available",
    "languages": ["Language 1", "Language 2"]
  },
  "summary": "extracted summary or a brief one you generate based on the resume",
  "skills": ["skill1", "skill2"],
  "experience": [
    {
      "title": "Job Title or Project Name",
      "company": "Company Name or 'Personal Project'/'Academic Project'",
      "duration": "Dates or 'Ongoing'",
      "bullets": ["bullet point 1", "bullet point 2"]
    }
  ],
  "education": [
    {
      "degree": "Degree/Diploma Name",
      "institution": "School/College Name",
      "duration": "Dates",
      "details": "GPA, honors, etc."
    }
  ],
  "certifications": ["Cert 1", "Cert 2"],
  "achievements": ["Achievement 1", "Achievement 2"],
  "strengths": ["Strength 1", "Strength 2"],
  "weaknesses": ["list of obvious weaknesses like lack of quantification, missing sections, etc."]
}

Rules:
1. Clean up any weird characters or markdown formatting from the raw text.
2. Ensure bullet points are captured exactly as written.
3. Look specifically for Name, Email, Phone, and LinkedIn in the header area.
"""

class AnalyzerAgent:
    def __init__(self, groq_client: GroqClient):
        self.groq = groq_client
        self.name = "AnalyzerAgent"

    async def run(self, resume_text: str) -> dict:
        print(f"[{self.name}] Analyzing resume...")
        user_message = f"Please analyze this resume text:\n\n{resume_text}"
        
        output = await self.groq.chat_json(
            system_prompt=ANALYZER_SYSTEM_PROMPT,
            user_message=user_message,
            temperature=0.1
        )
        return output
