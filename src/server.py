"""
FastAPI Server & Frontend Host for Placement Prep Knowledge Assistant
Exposes RESTful endpoints for RAG queries and serves the modern web interface.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import src.config as config
from src.agents.graph import run_query

app = FastAPI(
    title="Placement Prep Knowledge Assistant API",
    description="Multi-Agent RAG Backend with Hybrid Retrieval, Cross-Encoder Reranking, and Conflict Detection",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request & Response Schemas
class QueryRequest(BaseModel):
    query: str
    topic_filters: Optional[List[str]] = None
    company_filter: Optional[str] = None

class QueryResponse(BaseModel):
    query: str
    final_answer: str
    has_conflict: bool
    needs_synthesis: bool
    sources_used: List[str]
    retrieved_chunks: List[dict]

@app.get("/api/health")
def health_check():
    return {"status": "healthy", "service": "placement-prep-rag", "version": "1.0.0"}

@app.get("/api/stats")
def get_corpus_stats():
    """Returns corpus metrics: chunk count, senior count, company logs, and topic distribution."""
    chunks_path = Path(config.CHUNKS_FILE)
    if not chunks_path.exists():
        return {
            "total_chunks": 0, "total_seniors": 0, "total_companies": 0,
            "total_sources": 0, "companies_list": [], "topics_list": config.TOPICS
        }

    seniors, companies, sources = set(), set(), set()
    total_chunks = 0
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                c = json.loads(line)
                total_chunks += 1
                if c.get("senior_id"): seniors.add(c["senior_id"])
                if c.get("company"): companies.add(c["company"])
                if c.get("source_name"): sources.add(c["source_name"])

    return {
        "total_chunks": total_chunks,
        "total_seniors": len(seniors),
        "total_companies": len(companies),
        "total_sources": len(sources),
        "companies_list": sorted(list(companies)),
        "topics_list": config.TOPICS
    }

@app.post("/api/query", response_model=QueryResponse)
def handle_rag_query(request: QueryRequest):
    """Runs the multi-agent LangGraph workflow for a student placement query."""
    user_query = request.query.strip()
    if not user_query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        result = run_query(
            user_query=user_query,
            topic_filters=request.topic_filters,
            company_filter=request.company_filter
        )
        return QueryResponse(
            query=user_query,
            final_answer=result.get("final_answer", ""),
            has_conflict=result.get("has_conflict", False),
            needs_synthesis=result.get("needs_synthesis", False),
            sources_used=result.get("sources_used", []),
            retrieved_chunks=result.get("retrieved_chunks", [])
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")

# Mount frontend static files
FRONTEND_DIR = config.PROJECT_ROOT / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/")
    def serve_frontend():
        return FileResponse(str(FRONTEND_DIR / "index.html"))

if __name__ == "__main__":
    import uvicorn
    print("\n🚀 Starting Placement Prep Knowledge Assistant on http://localhost:8000")
    uvicorn.run("src.server:app", host="0.0.0.0", port=8000, reload=True)
