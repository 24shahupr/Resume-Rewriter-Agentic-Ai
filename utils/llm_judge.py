"""
utils/llm_judge.py

LLM-as-Judge framework.
Implements rubric-based evaluation + critique-revise loop for any agent output.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from utils.groq_client import GroqClient


# ─── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class RubricCriterion:
    name: str
    description: str
    weight: float          # 0.0–1.0, all weights in a rubric should sum to 1.0
    passing_threshold: int # minimum score (0–10) to not trigger a revision


@dataclass
class CriterionScore:
    criterion: str
    score: int             # 0–10
    rationale: str
    passed: bool


@dataclass
class JudgeVerdict:
    criterion_scores: List[CriterionScore]
    weighted_score: float       # 0–10 overall
    passed: bool
    critique: str               # actionable feedback if failed
    revision_instructions: str  # specific instructions for the reviser


# ─── Built-in Rubrics ─────────────────────────────────────────────────────────

REWRITER_RUBRIC: List[RubricCriterion] = [
    RubricCriterion(
        name="Keyword Integration",
        description="Resume naturally incorporates role-specific ATS keywords without keyword stuffing.",
        weight=0.20,
        passing_threshold=7,
    ),
    RubricCriterion(
        name="Action Verb Quality",
        description="Bullets open with strong, varied action verbs (Led, Engineered, Reduced, Deployed…). No passive voice.",
        weight=0.15,
        passing_threshold=7,
    ),
    RubricCriterion(
        name="Completeness & Accuracy",
        description="Must include candidate's Name, Contact Info, Education, and all original projects/experience. No major section should be omitted.",
        weight=0.30,
        passing_threshold=9,
    ),
    RubricCriterion(
        name="Quantification",
        description="Experience bullets use numbers, percentages, or placeholders like '[X]%' to indicate impact. Hallucinated metrics are penalized.",
        weight=0.15,
        passing_threshold=6,
    ),
    RubricCriterion(
        name="Role Alignment",
        description="Summary and skills section are clearly tailored to the target role, not generic.",
        weight=0.10,
        passing_threshold=7,
    ),
    RubricCriterion(
        name="Professional Tone",
        description="Language is concise, confident, and professional. No filler phrases or clichés.",
        weight=0.10,
        passing_threshold=7,
    ),
]

SCORER_RUBRIC: List[RubricCriterion] = [
    RubricCriterion(
        name="Scoring Accuracy",
        description="ATS/experience/structure scores reflect the actual resume quality vs. industry data. No inflation.",
        weight=0.35,
        passing_threshold=7,
    ),
    RubricCriterion(
        name="Gap Specificity",
        description="Identified gaps reference actual resume content and specific missing skills — not generic observations.",
        weight=0.30,
        passing_threshold=7,
    ),
    RubricCriterion(
        name="Recommendation Actionability",
        description="Each recommendation is concrete and directly actionable (not 'improve your bullets').",
        weight=0.35,
        passing_threshold=7,
    ),
]


# ─── Judge ────────────────────────────────────────────────────────────────────

JUDGE_SYSTEM_PROMPT = """You are a rigorous, impartial LLM judge.
You evaluate AI-generated resume content against a structured rubric.

For each criterion provided, score the output 0–10 and explain your reasoning concisely.
Then decide if the overall output passes (weighted_score >= 6.0).
If it fails, write a detailed critique and specific revision instructions.

Respond with ONLY valid JSON — no preamble, no markdown fences:

{
  "criterion_scores": [
    {
      "criterion": "<criterion name>",
      "score": <0-10>,
      "rationale": "<1-2 sentence explanation>",
      "passed": <true if score >= threshold>
    }
  ],
  "weighted_score": <float 0-10>,
  "passed": <true/false>,
  "critique": "<detailed critique if failed, empty string if passed>",
  "revision_instructions": "<specific numbered instructions for the reviser if failed, empty string if passed>"
}"""


class LLMJudge:
    """
    Evaluates agent outputs against a rubric using a separate LLM call.
    Supports critique-revise loops up to max_revisions.
    """

    MAX_REVISIONS = 3

    def __init__(self, groq_client: GroqClient):
        self.groq = groq_client

    async def evaluate(
        self,
        output_to_judge: Any,
        rubric: List[RubricCriterion],
        context: str = "",
    ) -> JudgeVerdict:
        """
        Evaluate an output against a rubric.
        Returns a JudgeVerdict with scores and optional critique.
        """
        rubric_text = "\n".join(
            f"- **{c.name}** (weight={c.weight:.0%}, pass>={c.passing_threshold}/10): {c.description}"
            for c in rubric
        )

        user_message = f"""## Context
{context}

## Output to Evaluate
{json.dumps(output_to_judge, indent=2) if isinstance(output_to_judge, dict) else str(output_to_judge)}

## Rubric Criteria
{rubric_text}

## Evaluation Task
Evaluate the output against each criterion. Score each from 0-10.
The output passes ONLY if the overall weighted_score is >= 6.0.

Return only the JSON verdict."""

        raw = await self.groq.chat_json(
            system_prompt=JUDGE_SYSTEM_PROMPT,
            user_message=user_message,
            temperature=0.1,
            max_tokens=1200,
        )

        # Build structured verdict
        criterion_scores = []
        raw_scores = raw.get("criterion_scores", [])
        if isinstance(raw_scores, list):
            for s in raw_scores:
                if isinstance(s, dict):
                    criterion_scores.append(
                        CriterionScore(
                            criterion=str(s.get("criterion", "Unknown")),
                            score=int(s.get("score", 0)),
                            rationale=str(s.get("rationale", "No rationale provided.")),
                            passed=bool(s.get("passed", False)),
                        )
                    )

        return JudgeVerdict(
            criterion_scores=criterion_scores,
            weighted_score=float(raw.get("weighted_score", 0)),
            passed=bool(raw.get("passed", False)),
            critique=str(raw.get("critique", "")),
            revision_instructions=str(raw.get("revision_instructions", "")),
        )

    async def judge_and_revise(
        self,
        initial_output: Any,
        rubric: List[RubricCriterion],
        reviser_fn,                  # async callable: (output, critique, instructions) -> new_output
        context: str = "",
        agent_name: str = "Agent",
    ) -> tuple[Any, JudgeVerdict, int]:
        """
        Full critique-revise loop:
        1. Judge initial output
        2. If failed → call reviser_fn with critique
        3. Re-judge revised output
        4. Repeat up to MAX_REVISIONS times
        Returns (final_output, final_verdict, revision_count)
        """
        output = initial_output
        revision_count = 0

        for attempt in range(self.MAX_REVISIONS + 1):
            verdict = await self.evaluate(output, rubric, context)

            label = "initial" if attempt == 0 else f"revision {attempt}"
            score_str = f"{verdict.weighted_score:.1f}/10"
            status = "✓ PASSED" if verdict.passed else "✗ FAILED"
            print(f"  [{agent_name}] Judge {label}: {score_str} — {status}")

            if verdict.passed or attempt == self.MAX_REVISIONS:
                break

            # Failed — trigger revision
            print(f"  [{agent_name}] Critique: {verdict.critique[:120]}...")
            print(f"  [{agent_name}] Requesting revision {attempt + 1}...")
            output = await reviser_fn(output, verdict.critique, verdict.revision_instructions)
            revision_count += 1

        return output, verdict, revision_count
