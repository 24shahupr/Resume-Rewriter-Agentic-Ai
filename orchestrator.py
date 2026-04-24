"""
orchestrator.py

Orchestrator Agent — Pipeline Controller
Controls the full multi-agent pipeline:

  [Analyzer] ──┐
               ├──► [Scorer] ──► [Rewriter] ──► Final Report
  [Researcher]─┘

Analyzer and Researcher run in PARALLEL (asyncio.gather).
Scorer waits for both. Rewriter waits for Scorer.

The Orchestrator also runs its own lightweight validation pass on the
final assembled report using a simple LLM-as-Judge check before returning.
"""

import asyncio
import time
from utils.groq_client import GroqClient
from utils.tavily_client import TavilyClient
from utils.llm_judge import LLMJudge, RubricCriterion
from agents.analyzer_agent import AnalyzerAgent
from agents.researcher_agent import ResearcherAgent
from agents.scorer_agent import ScorerAgent
from agents.rewriter_agent import RewriterAgent


# ─── Orchestrator-level pipeline rubric ──────────────────────────────────────
# Lightweight check: did the full pipeline produce a coherent, non-empty report?

PIPELINE_RUBRIC: list[RubricCriterion] = [
    RubricCriterion(
        name="Completeness",
        description="All four output sections (analysis, research, score, rewrite) are non-empty and structurally valid.",
        weight=0.40,
        passing_threshold=7,
    ),
    RubricCriterion(
        name="Internal Consistency",
        description="The rewritten resume addresses the gaps identified by the scorer. Projected score > original score.",
        weight=0.35,
        passing_threshold=6,
    ),
    RubricCriterion(
        name="Actionability",
        description="The final report gives the user clear next steps — improvement tips and skills to acquire are specific.",
        weight=0.25,
        passing_threshold=7,
    ),
]


class Orchestrator:
    """
    Central pipeline controller. Instantiates all 4 agents and coordinates execution.
    Exposes a single `run(resume_text, target_role)` coroutine.
    """

    def __init__(self, groq_api_key: str, tavily_api_key: str):
        self.groq = GroqClient(api_key=groq_api_key)
        self.tavily = TavilyClient(api_key=tavily_api_key)
        self.judge = LLMJudge(self.groq)

        # Instantiate all agents
        self.analyzer   = AnalyzerAgent(self.groq)
        self.researcher = ResearcherAgent(self.groq, self.tavily)
        self.scorer     = ScorerAgent(self.groq)
        self.rewriter   = RewriterAgent(self.groq)

    async def run_stream(self, resume_text: str, target_role: str):
        """
        Execute the full pipeline and yield progress steps as JSON strings.
        """
        pipeline_start = time.time()

        yield {"type": "step", "agent": "Orchestrator", "message": "Starting pipeline...", "step": 1}

        # ── Step 1: Validate input ────────────────────────────────────────────
        if len(resume_text.strip()) < 50:
            yield {"type": "error", "message": "Resume text is too short (< 50 characters)."}
            return

        # ── Step 2: Analyzer + Researcher in PARALLEL ─────────────────────────
        yield {"type": "step", "agent": "Analyzer/Researcher", "message": "Analyzing resume & researching market trends...", "step": 2}
        stage1_start = time.time()

        analyzer_output, researcher_output = await asyncio.gather(
            self.analyzer.run(resume_text),
            self.researcher.run(target_role),
        )

        # Basic validation: if analyzer failed to find anything, stop.
        if not analyzer_output.get("experience") and not analyzer_output.get("skills"):
            yield {"type": "error", "message": "Analyzer failed to parse resume content."}
            return

        # ── Step 3: Scorer ────────────────────────────────────────────────────
        yield {"type": "step", "agent": "Scorer", "message": "Scoring resume and identifying gaps...", "step": 3}
        stage2_start = time.time()
        scorer_output = await self.scorer.run(analyzer_output, researcher_output)

        # ── Step 4: Rewriter ──────────────────────────────────────────────────
        yield {"type": "step", "agent": "Rewriter", "message": "Rewriting and applying final judge review...", "step": 4}
        stage3_start = time.time()
        rewriter_output = await self.rewriter.run(
            analyzer_output, researcher_output, scorer_output
        )

        # ── Step 5: Assemble final report ─────────────────────────────────────
        total_time = time.time() - pipeline_start

        report = {
            "target_role": target_role,
            "raw_resume": resume_text,
            "pipeline_time_seconds": round(total_time, 2),
            "analysis": analyzer_output,
            "research": {
                "trending_skills":           researcher_output.get("trending_skills", []),
                "ats_keywords":              researcher_output.get("ats_keywords", []),
                "certifications_recommended":researcher_output.get("certifications_recommended", []),
                "role_responsibilities":     researcher_output.get("role_responsibilities", []),
                "offline_mode":              researcher_output.get("offline_mode", False),
            },
            "score": {
                "ats_score":        scorer_output.get("ats_score"),
                "experience_score": scorer_output.get("experience_score"),
                "structure_score":  scorer_output.get("structure_score"),
                "overall_score":    scorer_output.get("overall_score"),
                "gaps":             scorer_output.get("gaps", []),
                "strengths":        scorer_output.get("strengths", []),
                "recommendations":  scorer_output.get("recommendations", []),
                "_judge":           scorer_output.get("_judge", {}),
            },
            "rewritten": {
                "personal_info":    rewriter_output.get("personal_info", {}),
                "summary":          rewriter_output.get("rewritten_summary", ""),
                "experience":       rewriter_output.get("rewritten_experience", []),
                "education":        rewriter_output.get("education", []),
                "certifications":   rewriter_output.get("certifications", []),
                "achievements":     rewriter_output.get("achievements", []),
                "strengths":        rewriter_output.get("strengths", []),
                "skills":           rewriter_output.get("rewritten_skills", []),
                "skills_to_acquire":rewriter_output.get("skills_to_acquire", []),
                "improvement_tips": rewriter_output.get("improvement_tips", []),
                "projected_score":  rewriter_output.get("projected_score"),
                "_judge":           rewriter_output.get("_judge", {}),
            },
            "_pipeline_meta": {
                "score_before":         scorer_output.get("overall_score"),
                "score_after_projected":rewriter_output.get("projected_score"),
                "score_delta":          (
                    rewriter_output.get("projected_score", 0)
                    - scorer_output.get("overall_score", 0)
                ),
                "rewriter_judge_score": rewriter_output.get("_judge", {}).get("weighted_score"),
                "scorer_judge_score":   scorer_output.get("_judge", {}).get("weighted_score"),
                "total_revisions":      (
                    scorer_output.get("_judge", {}).get("revisions_made", 0)
                    + rewriter_output.get("_judge", {}).get("revisions_made", 0)
                ),
            },
        }

        # ── Step 6: Pipeline-level quality check (lightweight) ────────────────
        pipeline_verdict = await self.judge.evaluate(
            output_to_judge={
                "gaps_identified":    report["score"]["gaps"],
                "rewrite_summary":    report["rewritten"]["summary"],
                "improvement_tips":   report["rewritten"]["improvement_tips"],
                "skills_to_acquire":  report["rewritten"]["skills_to_acquire"],
                "score_delta":        report["_pipeline_meta"]["score_delta"],
            },
            rubric=PIPELINE_RUBRIC,
            context=f"Target role: {target_role}",
        )

        report["_pipeline_meta"]["pipeline_quality_score"] = pipeline_verdict.weighted_score
        report["_pipeline_meta"]["pipeline_quality_passed"] = pipeline_verdict.passed

        yield {"type": "complete", "report": report}

    async def run(self, resume_text: str, target_role: str) -> dict:
        """
        Execute the full pipeline and return the assembled report.
        """
        pipeline_start = time.time()

        print("\n" + "═" * 60)
        print(f"  ORCHESTRATOR — Starting pipeline")
        print(f"  Target Role : {target_role}")
        print(f"  Resume size : {len(resume_text)} characters")
        print("═" * 60)

        # ── Step 1: Validate input ────────────────────────────────────────────
        if len(resume_text.strip()) < 50:
            raise ValueError("Resume text is too short (< 50 characters). Please provide full resume.")

        # ── Step 2: Analyzer + Researcher in PARALLEL ─────────────────────────
        print("\n[Orchestrator] ▶ Stage 1 — Running Analyzer & Researcher in parallel...")
        stage1_start = time.time()

        analyzer_output, researcher_output = await asyncio.gather(
            self.analyzer.run(resume_text),
            self.researcher.run(target_role),
        )

        print(f"[Orchestrator] ✓ Stage 1 complete in {time.time() - stage1_start:.1f}s")

        # Basic validation: if analyzer failed to find anything, stop.
        if not analyzer_output.get("experience") and not analyzer_output.get("skills"):
            print("[Orchestrator] ✗ Critical Error: Analyzer failed to parse resume content. Stopping.")
            return {
                "error": "Failed to parse resume content. Please ensure the resume text is readable.",
                "analyzer_raw": analyzer_output
            }

        # ── Step 3: Scorer ────────────────────────────────────────────────────
        print("\n[Orchestrator] ▶ Stage 2 — Running Scorer (with Judge loop)...")
        stage2_start = time.time()

        scorer_output = await self.scorer.run(analyzer_output, researcher_output)

        print(f"[Orchestrator] ✓ Stage 2 complete in {time.time() - stage2_start:.1f}s")

        # ── Step 4: Rewriter ──────────────────────────────────────────────────
        print("\n[Orchestrator] ▶ Stage 3 — Running Rewriter (with Judge loop)...")
        stage3_start = time.time()

        rewriter_output = await self.rewriter.run(
            analyzer_output, researcher_output, scorer_output
        )

        print(f"[Orchestrator] ✓ Stage 3 complete in {time.time() - stage3_start:.1f}s")

        # ── Step 5: Assemble final report ─────────────────────────────────────
        total_time = time.time() - pipeline_start

        report = {
            "target_role": target_role,
            "raw_resume": resume_text,
            "pipeline_time_seconds": round(total_time, 2),
            "analysis": analyzer_output,
            "research": {
                "trending_skills":           researcher_output.get("trending_skills", []),
                "ats_keywords":              researcher_output.get("ats_keywords", []),
                "certifications_recommended":researcher_output.get("certifications_recommended", []),
                "role_responsibilities":     researcher_output.get("role_responsibilities", []),
                "offline_mode":              researcher_output.get("offline_mode", False),
            },
            "score": {
                "ats_score":        scorer_output.get("ats_score"),
                "experience_score": scorer_output.get("experience_score"),
                "structure_score":  scorer_output.get("structure_score"),
                "overall_score":    scorer_output.get("overall_score"),
                "gaps":             scorer_output.get("gaps", []),
                "strengths":        scorer_output.get("strengths", []),
                "recommendations":  scorer_output.get("recommendations", []),
                "_judge":           scorer_output.get("_judge", {}),
            },
            "rewritten": {
                "personal_info":    rewriter_output.get("personal_info", {}),
                "summary":          rewriter_output.get("rewritten_summary", ""),
                "experience":       rewriter_output.get("rewritten_experience", []),
                "education":        rewriter_output.get("education", []),
                "certifications":   rewriter_output.get("certifications", []),
                "achievements":     rewriter_output.get("achievements", []),
                "skills":           rewriter_output.get("rewritten_skills", []),
                "skills_to_acquire":rewriter_output.get("skills_to_acquire", []),
                "improvement_tips": rewriter_output.get("improvement_tips", []),
                "projected_score":  rewriter_output.get("projected_score"),
                "_judge":           rewriter_output.get("_judge", {}),
            },
            "_pipeline_meta": {
                "score_before":         scorer_output.get("overall_score"),
                "score_after_projected":rewriter_output.get("projected_score"),
                "score_delta":          (
                    rewriter_output.get("projected_score", 0)
                    - scorer_output.get("overall_score", 0)
                ),
                "rewriter_judge_score": rewriter_output.get("_judge", {}).get("weighted_score"),
                "scorer_judge_score":   scorer_output.get("_judge", {}).get("weighted_score"),
                "total_revisions":      (
                    scorer_output.get("_judge", {}).get("revisions_made", 0)
                    + rewriter_output.get("_judge", {}).get("revisions_made", 0)
                ),
            },
        }

        # ── Step 6: Pipeline-level quality check (lightweight) ────────────────
        print("\n[Orchestrator] ▶ Final pipeline quality check...")
        pipeline_verdict = await self.judge.evaluate(
            output_to_judge={
                "gaps_identified":    report["score"]["gaps"],
                "rewrite_summary":    report["rewritten"]["summary"],
                "improvement_tips":   report["rewritten"]["improvement_tips"],
                "skills_to_acquire":  report["rewritten"]["skills_to_acquire"],
                "score_delta":        report["_pipeline_meta"]["score_delta"],
            },
            rubric=PIPELINE_RUBRIC,
            context=f"Target role: {target_role}",
        )

        report["_pipeline_meta"]["pipeline_quality_score"] = pipeline_verdict.weighted_score
        report["_pipeline_meta"]["pipeline_quality_passed"] = pipeline_verdict.passed

        print(
            f"\n{'═'*60}\n"
            f"  PIPELINE COMPLETE\n"
            f"  Total time          : {total_time:.1f}s\n"
            f"  ATS score before    : {report['_pipeline_meta']['score_before']}/100\n"
            f"  ATS score projected : {report['_pipeline_meta']['score_after_projected']}/100\n"
            f"  Score delta         : +{report['_pipeline_meta']['score_delta']} pts\n"
            f"  Total LLM revisions : {report['_pipeline_meta']['total_revisions']}\n"
            f"  Pipeline quality    : {pipeline_verdict.weighted_score:.1f}/10 "
            f"({'✓' if pipeline_verdict.passed else '✗'})\n"
            f"{'═'*60}\n"
        )

        return report