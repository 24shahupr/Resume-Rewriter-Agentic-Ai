from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import json
from dotenv import load_dotenv
import asyncio
import sys

# Add the parent directory to sys.path so we can import from agents and utils
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from orchestrator import Orchestrator

load_dotenv()

app = FastAPI()

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ResumeRequest(BaseModel):
    resume_text: str
    target_role: str
    groq_api_key: str = None
    tavily_api_key: str = None

@app.get("/")
async def serve_index():
    index_path = os.path.join(ROOT_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"error": "index.html not found"}

@app.get("/api/health")
def health_check():
    return {"status": "healthy"}

@app.get("/api/run")
async def run_pipeline_stream(
    resume: str = Query(...),
    role: str = Query(...),
    groq_api_key: str = Query(None),
    tavily_api_key: str = Query(None)
):
    groq_key = groq_api_key or os.getenv("GROQ_API_KEY")
    tavily_key = tavily_api_key or os.getenv("TAVILY_API_KEY")

    if not groq_key or not tavily_key:
        # SSE error format
        async def error_gen():
            yield f"data: {json.dumps({'type': 'error', 'message': 'API keys are required.'})}\n\n"
        return StreamingResponse(error_gen(), media_type="text/event-stream")

    orchestrator = Orchestrator(groq_api_key=groq_key, tavily_api_key=tavily_key)

    async def event_generator():
        try:
            async for step in orchestrator.run_stream(resume, role):
                yield f"data: {json.dumps(step)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/api/run")
async def run_pipeline(request: ResumeRequest):
    groq_key = request.groq_api_key or os.getenv("GROQ_API_KEY")
    tavily_key = request.tavily_api_key or os.getenv("TAVILY_API_KEY")

    if not groq_key or not tavily_key:
        raise HTTPException(status_code=400, detail="API keys are required.")

    try:
        orchestrator = Orchestrator(groq_api_key=groq_key, tavily_api_key=tavily_key)
        report = await orchestrator.run(request.resume_text, request.target_role)
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# For local development
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
