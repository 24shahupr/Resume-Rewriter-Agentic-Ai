# 🧩 Task Decomposition — ResumeAgent AI

The system is architected as a **multi-agent orchestration pipeline**. Each task is isolated to a specific agent, ensuring modularity, reliability, and superior quality control through iterative loops.

---

## 1. Phase: Data Ingestion & Environmental Context
**Goal**: Build a foundation of structured candidate data and real-world market requirements.

### 🔍 Analyzer Agent (Extraction)
- **Input**: Raw text from user's current resume.
- **Process**:
    - Identifies Personal Info (Name, Contact, LinkedIn).
    - Structures Experience & Projects into bulleted lists.
    - Extracts Education, Certifications, and Strengths.
- **Output**: A comprehensive JSON representation of the candidate's profile.

### 🌐 Researcher Agent (Web Search)
- **Input**: Target job role (e.g., "Frontend Developer").
- **Process**:
    - Executes **Tavily Search** to find top 2026 hiring trends.
    - Identifies high-frequency ATS keywords.
    - Scans for required technical and soft skills for the specific role.
- **Output**: A market trend report used to guide the scoring and rewriting phases.

---

## 2. Phase: Evaluation & Gap Analysis
**Goal**: Critically assess the current resume against the market research.

### 📊 Scorer Agent (The Judge)
- **Input**: Analyzer Output + Researcher Output.
- **Process**:
    - Uses **LLM-as-Judge** with a custom rubric.
    - Assigns an ATS compatibility score (0-100).
    - Identifies specific "Gaps" (e.g., "Missing React hooks experience", "Weak action verbs").
- **Output**: A score report with actionable improvement tips.

---

## 3. Phase: Transformation & Quality Assurance
**Goal**: Execute the rewrite and ensure professional integrity.

### ✍️ Rewriter Agent (The Architect)
- **Input**: All previous agent outputs.
- **Process**:
    - Rewrites the professional summary for maximum impact.
    - Enhances experience bullets using the **Action Verb + Task + Result** formula.
    - Groups skills into logical categories.
- **Critique-Revise Loop**:
    - After generating the first draft, the **LLM Judge** evaluates it for hallucinations and accuracy.
    - If inaccuracies are found, the Rewriter receives specific instructions to fix them.
- **Output**: The finalized, polished resume data.

---

## 4. Phase: Orchestration & Delivery
**Goal**: Manage the lifecycle and stream updates to the user.

### 📦 Orchestrator (The Conductor)
- **Process**:
    - Manages `asyncio` execution of agents (Parallel Analyzer & Researcher).
    - Handles state management between stages.
    - **SSE Streaming**: Sends real-time progress updates (e.g., "Scoring resume...") to the frontend via Server-Sent Events.
- **Output**: The complete JSON report ready for UI rendering and TXT download.
