"""
agents/scorer_agent.py

Agent 3 — ATS Scorer & Gap Analyzer
Compares parsed resume against industry research to produce an ATS compatibility
score and detailed gap analysis.

LLM-as-Judge Integration:
  After generating the initial score report, a judge LLM evaluates it against
  SCORER_RUBRIC (scoring accuracy, gap specificity, recommendation actionability).
  If the verdict fails, a reviser prompt re-generates the report with the critique
  attached — up to 2 revision cycles.
"""

import json
from utils.groq_client import GroqClient
from utils.llm_judge import LLMJudge, SCORER_RUBRIC


# ─── Prompts ──────────────────────────────────────────────────────────────────

SCORER_SYSTEM_PROMPT = """You are an ATS (Applicant Tracking System) expert and senior career strategist.

You receive:
1. Parsed resume data  — skills, experience bullets, sections present, weaknesses
2. Industry research   — trending skills, ATS keywords, role responsibilities

Your job: produce an honest, specific scoring report. No inflation. If a resume is weak, say so.

Return EXACTLY this JSON (no other text):
{
  "thoughts": "<brief reasoning about the scores and findings>",
  "ats_score": <0-100, calculate based on keyword overlap>,
  "experience_score": <0-100, calculate based on bullet quality>,
  "structure_score": <0-100, calculate based on layout/sections>,
  "overall_score": <0-100, weighted: ats*0.45 + exp*0.35 + struct*0.20>,
  "gaps": [
    "<specific gap referencing actual resume content + why it matters for role>"
  ],
  "strengths": [
    "<specific strength referencing actual resume content>"
  ],
  "recommendations": [
    "<numbered, concrete, actionable recommendation — not generic tips>"
  ]
}

Rules:
- gaps and recommendations must reference ACTUAL skills/bullets from the resume.
- Every recommendation must be a specific action, not vague advice.
- ATS score = what % of industry ATS keywords appear in the resume.
- Experience score = quality of bullet writing (action verbs, metrics, specificity). If the candidate is a student, evaluate their PROJECTS with the same weight as work experience.
- Structure score = completeness of sections (Summary, Education, Experience/Projects, Skills).
- Be objective: if a student has no work experience but strong projects, do not penalize the experience score heavily. Focus on their technical skills, project complexity, and results.
- Ensure the 'gaps' section identifies missing skills or metrics, NOT just 'lack of work experience' if they have projects.
"""


REVISER_SYSTEM_PROMPT = """You are a resume scoring specialist. You previously generated a resume score report
that was evaluated by a judge and found lacking. 

You must revise the score report based on the critique and instructions provided.
Return the corrected JSON in the EXACT same schema as before. Raw JSON only."""


# ─── Agent ────────────────────────────────────────────────────────────────────

class ScorerAgent:
    """
    Agent 3: Scores resume ATS compatibility + gap analysis.
    Uses LLM-as-Judge with SCORER_RUBRIC + critique-revise loop.

    Input:  analyzer_output (dict), researcher_output (dict)
    Output: scored dict with judge verdict metadata attached
    """

    def __init__(self, groq_client: GroqClient):
        self.groq = groq_client
        self.judge = LLMJudge(groq_client)
        self.name = "ScorerAgent"

    # ── Initial scoring ───────────────────────────────────────────────────────

    async def _generate_score(
        self,
        analyzer_output: dict,
        researcher_output: dict,
        critique: str = "",
        revision_instructions: str = "",
    ) -> dict:
        """Generate (or revise) the score report."""
        base_message = f"""## Resume Analysis
{json.dumps(analyzer_output, indent=2)}

## Industry Research
{json.dumps(researcher_output, indent=2)}"""

        if critique:
            user_message = f"""{base_message}

## Previous Output Was Rejected — Judge Critique
{critique}

## Revision Instructions
{revision_instructions}

Produce a corrected score report following all instructions above."""
            system = REVISER_SYSTEM_PROMPT
        else:
            user_message = base_message + "\n\nScore this resume against the industry data."
            system = SCORER_SYSTEM_PROMPT

        return await self.groq.chat_json(
            system_prompt=system,
            user_message=user_message,
            temperature=0.1,
            max_tokens=1200,
        )

    # ── Reviser callable for judge loop ──────────────────────────────────────

    def _make_reviser(self, analyzer_output: dict, researcher_output: dict):
        async def reviser(prev_output: dict, critique: str, instructions: str) -> dict:
            return await self._generate_score(
                analyzer_output, researcher_output, critique, instructions
            )
        return reviser

    # ── Main run ──────────────────────────────────────────────────────────────

    async def run(self, analyzer_output: dict, researcher_output: dict) -> dict:
        print(f"[{self.name}] Generating ATS score report...")

        # Step 1 — initial generation
        initial_score = await self._generate_score(analyzer_output, researcher_output)

        # Step 2 — judge + critique-revise loop
        judge_context = (
            f"Target role skills needed: {researcher_output.get('trending_skills', []) if isinstance(researcher_output, dict) else []}\n"
            f"Resume skills present: {analyzer_output.get('skills', []) if isinstance(analyzer_output, dict) else []}"
        )

        final_score, verdict, revisions = await self.judge.judge_and_revise(
            initial_output=initial_score,
            rubric=SCORER_RUBRIC,
            reviser_fn=self._make_reviser(analyzer_output, researcher_output),
            context=judge_context,
            agent_name=self.name,
        )

        # Ensure final_score is a dict before attaching metadata
        if not isinstance(final_score, dict):
            print(f"[{self.name}] Warning: Final score is not a dict. Wrapping in dict.")
            final_score = {"raw_output": final_score}

        # Attach judge metadata
        final_score["_judge"] = {
            "weighted_score": verdict.weighted_score,
            "passed": verdict.passed,
            "revisions_made": revisions,
            "criterion_scores": [
                {
                    "criterion": cs.criterion,
                    "score": cs.score,
                    "passed": cs.passed,
                }
                for cs in verdict.criterion_scores
            ],
        }

        print(
            f"[{self.name}] ✓ Scoring complete. "
            f"Overall ATS: {final_score.get('overall_score', '?')}/100 | "
            f"Judge: {verdict.weighted_score:.1f}/10 | "
            f"Revisions: {revisions}"
        )
        return final_score
