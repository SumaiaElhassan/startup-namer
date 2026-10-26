"""
main.py
FastAPI server: serves the frontend and exposes POST /api/generate.
(Boilerplate routing - AI-assisted)
"""
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.llm_client import generate_names
from app.nlp_extractor import extract_entities
from app.schemas import GenerateResponse, IdeaRequest, NLPEntities

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

app = FastAPI(title="Startup Namer")


@app.post("/api/generate", response_model=GenerateResponse)
def generate(req: IdeaRequest):
    """Pipeline: classical NLP extraction -> LLM naming -> structured JSON."""
    nlp = extract_entities(req.idea)
    try:
        suggestions = generate_names(req.idea, nlp)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM error: {e}")
    return GenerateResponse(nlp=NLPEntities(**nlp), suggestions=suggestions)


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")