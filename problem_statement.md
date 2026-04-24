# Problem Statement

## The Challenge
Job seekers often struggle to align their resumes with the specific requirements of modern Applicant Tracking Systems (ATS) and rapidly changing industry trends. A generic resume often fails to pass initial automated screenings, even if the candidate is highly qualified.

Traditional resume builders are static and don't provide:
1.  **Contextual Awareness**: Understanding what skills are currently trending for a specific role *today*.
2.  **Qualitative Feedback**: Deep analysis of *how* experience is described (e.g., lack of quantification or weak action verbs).
3.  **Tailored Rewriting**: Automatically adjusting language to match the target role's specific keywords and responsibilities.

## The Solution: Agentic Resume Optimization
We propose a multi-agent AI system that automates the resume optimization pipeline. Instead of a single prompt, which often lacks depth and accuracy, we use a specialized team of agents:

-   **Researcher**: Discovers real-time market needs.
-   **Analyzer**: Understands the starting point.
-   **Scorer**: Provides a "tough love" evaluation based on the research.
-   **Rewriter**: Executes the improvements based on the score and research.

By using an **LLM-as-Judge** pattern, the system self-corrects, ensuring that the final output is not just "different," but objectively better and aligned with professional standards.
