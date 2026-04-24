# 📝 ResumeAgent AI — Professional Multi-Agent Pipeline

A powerful **multi-agent workflow** that transforms raw resumes into high-impact, ATS-optimized professional documents. Each stage is handled by a specialized agent (Analyzer, Researcher, Scorer, Rewriter) coordinated by an Orchestrator with built-in **LLM-as-Judge** quality control.

The app features a **FastAPI** backend designed for **Vercel** serverless deployment and a modern **SSE (Server-Sent Events)** powered frontend for real-time progress tracking.

## � How the pipeline works

| Stage | Role | What happens |
|--------|------|----------------|
| 1 | **Analyzer** | Deep-parses raw text into structured JSON. Extracts skills, experience, and projects while identifying initial weaknesses. |
| 2 | **Researcher** | Conducts real-time market research via **Tavily Search** to find trending skills and ATS keywords for the target role. |
| 3 | **Scorer** | Performs a "tough love" gap analysis. Uses **LLM-as-Judge** to evaluate the resume against research data and assign an ATS score. |
| 4 | **Rewriter** | Multi-turn transformation. Rewrites bullets with action verbs and quantification. Includes a critique-revise loop to ensure **100% data integrity** (no hallucinations). |

## 🛠️ Tech stack

- **Python 3.10+**
- **FastAPI** — High-performance backend with SSE support.
- **Groq SDK** — Powered by `llama-3.1-8b-instant` for ultra-fast agentic reasoning.
- **Tavily API** — Real-time web search for industry-specific market trends.
- **Vanilla JS / CSS** — Modern, responsive frontend with pulse animations and real-time logs.
- **Vercel** — Optimized for serverless deployment.

## ⚡ Quick start

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd resume-agent
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   Create a `.env` file in the root:
   ```env
   GROQ_API_KEY=your_groq_key
   TAVILY_API_KEY=your_tavily_key
   GROQ_MODEL=llama-3.1-8b-instant
   ```

4. **Run the app**
   ```bash
   # Start the FastAPI backend
   python api/index.py
   ```
   Open `index.html` in your browser (via a local server or directly).

## 📂 Project layout

```text
.
├── api/                   # FastAPI backend (Vercel serverless functions)
│   └── index.py           # Main API entry point & SSE logic
├── agents/                # Specialized AI Agents
│   ├── analyzer_agent.py  # Resume parsing & extraction
│   ├── researcher_agent.py# Market trend research (Tavily)
│   ├── scorer_agent.py    # ATS scoring & gap analysis
│   └── rewriter_agent.py  # Professional resume transformation
├── utils/                 # Core utilities
│   ├── groq_client.py     # Robust Groq API wrapper with retries
│   ├── tavily_client.py   # Async search client
│   └── llm_judge.py       # LLM-as-Judge framework (Critique & Revise)
├── index.html             # Professional UI with real-time pipeline tracking
├── orchestrator.py        # Central pipeline controller
├── vercel.json            # Vercel deployment configuration
└── requirements.txt       # Project dependencies
```

## � Security & Accuracy

- **Data Integrity**: The Rewriter Agent is governed by strict rules to prevent hallucinations. It *never* adds fake certifications or skills.
- **Privacy**: No resume data is stored permanently. The pipeline runs in-memory and returns the result to the user.
- **Reliability**: Built-in exponential backoff for Groq API rate limits (429 errors).
