import json
from utils.groq_client import GroqClient
from utils.tavily_client import TavilyClient

RESEARCHER_SYSTEM_PROMPT = """You are a Market Research Agent specializing in career trends and ATS (Applicant Tracking System) optimization.
Your task is to take search results about a specific job role and extract trending skills, key ATS keywords, and typical responsibilities.

Return EXACTLY this JSON:
{
  "trending_skills": ["skill1", "skill2"],
  "ats_keywords": ["keyword1", "keyword2"],
  "certifications_recommended": ["cert1", "cert2"],
  "role_responsibilities": ["resp1", "resp2"]
}
"""

class ResearcherAgent:
    def __init__(self, groq_client: GroqClient, tavily_client: TavilyClient):
        self.groq = groq_client
        self.tavily = tavily_client
        self.name = "ResearcherAgent"

    async def run(self, target_role: str) -> dict:
        print(f"[{self.name}] Researching market trends for: {target_role}...")
        
        # 1. Search for trends
        search_query = f"current skills and ATS keywords for {target_role} 2024 2025"
        search_context = await self.tavily.search(search_query)
        
        # 2. Process with LLM
        user_message = f"Based on these search results, identify trends for the role '{target_role}':\n\n{search_context}"
        
        output = await self.groq.chat_json(
            system_prompt=RESEARCHER_SYSTEM_PROMPT,
            user_message=user_message,
            temperature=0.1
        )
        
        # Add a flag if no search results were found (offline mode simulation or error)
        if not search_context or "No search results" in search_context:
            output["offline_mode"] = True
        else:
            output["offline_mode"] = False
            
        return output
