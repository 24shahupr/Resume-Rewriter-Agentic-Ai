"""
agents/rewriter_agent.py

Agent 4 — Resume Rewriter
Takes all prior agent outputs and rewrites the resume with:
  - Role-tailored professional summary
  - Upgraded experience bullets (action verbs + quantification)
  - Optimized skills section with gap-filling recommendations
  - Specific improvement tips

LLM-as-Judge Integration (deepest loop in the pipeline):
  REWRITER_RUBRIC has 5 criteria covering keyword integration, action verbs,
  quantification, role alignment, and professional tone.
  After each generation the judge scores it. Failure triggers a critique-revise
  cycle (up to 2 rounds) where the reviser receives the full critique + numbered
  instructions and must address each point explicitly.
"""

import json
from utils.groq_client import GroqClient
from utils.llm_judge import LLMJudge, REWRITER_RUBRIC


# ─── Prompts ──────────────────────────────────────────────────────────────────

REWRITER_SYSTEM_PROMPT = """You are a world-class professional resume writer. 
Your goal is to REWRITE the candidate's existing resume to be more professional and impactful.
You are a REWRITER, not a generator. You must be 100% factually accurate.

STRICT DATA INTEGRITY RULES:
1. NO HALLUCINATIONS: Do NOT add certifications, degrees, or awards that are not in the original.
2. NO NEW SKILLS: Do NOT add technical skills (e.g., AWS, Docker, Node.js) if they are not in the original. Put recommended skills in 'skills_to_acquire' ONLY.
3. PRESERVE ALL SECTIONS: You MUST include a Summary, Skills, Experience (including all Projects), Education, Certifications, Achievements, Strengths, and Personal Details (DOB, Languages).
4. PROJECTS ARE EXPERIENCE: Do NOT summarize projects into the achievements section. Every project from the original must be a separate entry in 'rewritten_experience'.
5. PERSONAL INFO: You MUST include the candidate's name, email, phone, location, linkedin, dob, and languages.

STRICT FORMATTING RULES:
- Rewritten summary: 3-4 lines max, professional, keyword-rich using ONLY original facts.
- Rewritten experience: Keep original titles/companies. Use strong action verbs. Use [X]% or [Number] for metrics if the original doesn't have them.
- Skills: Group original skills into categories (e.g., Languages, Frameworks, Tools).

Return EXACTLY this JSON:
{
  "thoughts": "<reasoning about the improvements made while staying 100% faithful to the original data>",
  "personal_info": {
    "name": "<original name>",
    "location": "<original location>",
    "email": "<original email>",
    "phone": "<original phone>",
    "linkedin": "<original linkedin>",
    "dob": "<original DOB>",
    "languages": ["<original languages>"]
  },
  "rewritten_summary": "<professional summary>",
  "rewritten_experience": [
    {
      "title": "<original title>",
      "company": "<original company>",
      "duration": "<original dates>",
      "bullets": [
        "<Impactful bullet point using action verbs and [X]% placeholders.>",
        "..."
      ]
    }
  ],
  "education": [
    {
      "degree": "<original degree>",
      "institution": "<original institution>",
      "duration": "<original duration>",
      "details": "<original details>"
    }
  ],
  "certifications": ["<original certs only>"],
  "achievements": ["<original achievements only>"],
  "strengths": ["<original strengths only>"],
  "rewritten_skills": ["<original skills grouped and optimized>"],
  "skills_to_acquire": ["<recommended skills from research>"],
  "improvement_tips": ["<tips for the candidate>"],
  "projected_score": <estimated score 0-100>
}
"""


REVISER_SYSTEM_PROMPT = """You are a professional resume writer. Your previous rewrite was evaluated by a
quality judge and did not meet the required standard.

You must revise the rewritten resume based on the critique and instructions provided.
Address EVERY point in the revision instructions explicitly.
Return corrected JSON in the EXACT same schema. Raw JSON only, no markdown."""


# ─── Agent ────────────────────────────────────────────────────────────────────

class RewriterAgent:
    """
    Agent 4: Rewrites resume content using all prior agent outputs.
    Uses LLM-as-Judge with REWRITER_RUBRIC + up-to-2 critique-revise loops.

    Input:  analyzer_output, researcher_output, scorer_output (dicts)
    Output: rewritten resume dict with judge metadata
    """

    def __init__(self, groq_client: GroqClient):
        self.groq = groq_client
        self.judge = LLMJudge(groq_client)
        self.name = "RewriterAgent"

    # ── Generation ────────────────────────────────────────────────────────────

    async def _generate_rewrite(
        self,
        analyzer_output: dict,
        researcher_output: dict,
        scorer_output: dict,
        critique: str = "",
        revision_instructions: str = "",
    ) -> dict:
        """Generate (or revise) the rewritten resume."""

        base_message = f"""## Original Resume (Parsed)
{json.dumps(analyzer_output, indent=2)}

## Industry Research for Target Role
{json.dumps(researcher_output, indent=2)}

## ATS Score Report & Gaps
{json.dumps(scorer_output, indent=2)}"""

        if critique:
            user_message = f"""{base_message}

## ⚠ Previous Rewrite Was Rejected by Quality Judge
### Critique
{critique}

### Revision Instructions — Address ALL of these:
{revision_instructions}

Produce a revised rewrite that explicitly addresses every instruction above."""
            system = REVISER_SYSTEM_PROMPT
        else:
            user_message = base_message + "\n\nProduce the rewritten resume now."
            system = REWRITER_SYSTEM_PROMPT

        return await self.groq.chat_json(
            system_prompt=system,
            user_message=user_message,
            temperature=0.3,   # slightly higher for creative rewriting
            max_tokens=2000,
        )

    # ── Reviser callable ──────────────────────────────────────────────────────

    def _make_reviser(
        self,
        analyzer_output: dict,
        researcher_output: dict,
        scorer_output: dict,
    ):
        async def reviser(prev_output: dict, critique: str, instructions: str) -> dict:
            return await self._generate_rewrite(
                analyzer_output,
                researcher_output,
                scorer_output,
                critique,
                instructions,
            )
        return reviser

    # ── Main run ──────────────────────────────────────────────────────────────

    async def run(
        self,
        analyzer_output: dict,
        researcher_output: dict,
        scorer_output: dict,
    ) -> dict:
        print(f"[{self.name}] Generating rewritten resume...")

        # Step 1 — initial rewrite
        initial_rewrite = await self._generate_rewrite(
            analyzer_output, researcher_output, scorer_output
        )

        # Step 2 — judge context (give judge the industry keywords to check against)
        judge_context = (
            f"Target role ATS keywords: {researcher_output.get('ats_keywords', []) if isinstance(researcher_output, dict) else []}\n"
            f"Trending skills needed: {researcher_output.get('trending_skills', []) if isinstance(researcher_output, dict) else []}\n"
            f"Identified gaps to address: {scorer_output.get('gaps', []) if isinstance(scorer_output, dict) else []}\n"
            f"Original experience bullets (for comparison): "
            + json.dumps([
                e.get("bullets", [])
                for e in (analyzer_output.get("experience", []) if isinstance(analyzer_output, dict) else [])
            ])
        )

        # Step 3 — judge + critique-revise loop
        final_rewrite, verdict, revisions = await self.judge.judge_and_revise(
            initial_output=initial_rewrite,
            rubric=REWRITER_RUBRIC,
            reviser_fn=self._make_reviser(analyzer_output, researcher_output, scorer_output),
            context=judge_context,
            agent_name=self.name,
        )

        # Ensure final_rewrite is a dict before attaching metadata
        if not isinstance(final_rewrite, dict):
            print(f"[{self.name}] Warning: Final rewrite is not a dict. Wrapping in dict.")
            final_rewrite = {"raw_output": final_rewrite}

        # Attach judge metadata
        final_rewrite["_judge"] = {
            "weighted_score": verdict.weighted_score,
            "passed": verdict.passed,
            "revisions_made": revisions,
            "criterion_scores": [
                {
                    "criterion": cs.criterion,
                    "score": cs.score,
                    "rationale": cs.rationale,
                    "passed": cs.passed,
                }
                for cs in verdict.criterion_scores
            ],
            "final_critique": verdict.critique if not verdict.passed else "",
        }

        bullet_count = sum(
            len(e.get("bullets", []))
            for e in final_rewrite.get("rewritten_experience", [])
        )
        print(
            f"[{self.name}] ✓ Rewrite complete. "
            f"{bullet_count} bullets rewritten | "
            f"Projected score: {final_rewrite.get('projected_score', '?')}/100 | "
            f"Judge: {verdict.weighted_score:.1f}/10 | "
            f"Revisions: {revisions}"
        )
        return final_rewrite
