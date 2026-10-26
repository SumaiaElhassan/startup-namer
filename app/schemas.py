"""
schemas.py
Pydantic models that define (and validate) every data shape in the app.
"""
from pydantic import BaseModel, ConfigDict, Field


class NameSuggestion(BaseModel):
    """One name + a one-sentence rationale."""
    model_config = ConfigDict(extra="forbid")
    name: str
    rationale: str


class NamingResult(BaseModel):
    """The exact JSON structure the LLM must return."""
    model_config = ConfigDict(extra="forbid")
    legal_names: list[NameSuggestion]
    product_names: list[NameSuggestion]


class IdeaRequest(BaseModel):
    """What the frontend sends: the user's idea (max 300 characters)."""
    idea: str = Field(..., min_length=10, max_length=300)


class NLPEntities(BaseModel):
    """Output of the classical NLP step."""
    language: str
    keywords: list[str]
    entities: list[dict]
    noun_chunks: list[str]


class GenerateResponse(BaseModel):
    """Final API response: NLP tags + LLM name suggestions."""
    nlp: NLPEntities
    suggestions: NamingResult