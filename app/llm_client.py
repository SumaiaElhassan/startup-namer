"""
llm_client.py
Sends the idea + extracted NLP entities to Groq and returns validated name suggestions.
JSON is enforced in two layers:
  1) Groq Structured Outputs (json_schema, strict mode)
  2) Pydantic validation + retry
"""
import json
import os

from dotenv import load_dotenv
from groq import Groq
from pydantic import ValidationError

from app.schemas import NamingResult

load_dotenv()  # read GROQ_API_KEY from the .env file

MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SYSTEM_PROMPT = """You are an expert startup branding consultant.
You receive a startup idea and keywords extracted from it by an NLP pipeline.

Rules:
1. legal_names: exactly 3 formal, descriptive company names suitable for official
   registration. English names must end with "LLC". Arabic names must start with "شركة".
2. product_names: exactly 3 catchy, innovative product names. Each is a single word
   or a creative compound/coined word, written in the SAME LANGUAGE as the idea.
3. Every rationale is ONE short sentence, in the same language as the idea.
4. Ground your suggestions in the provided NLP keywords and entities.
Respond with JSON only."""


def build_user_prompt(idea: str, nlp: dict) -> str:
    """Combine the original text with the NLP output into one prompt."""
    return (
        f"Startup idea:\n{idea}\n\n"
        f"Detected language: {nlp['language']}\n"
        f"Keywords: {', '.join(nlp['keywords']) or 'none'}\n"
        f"Noun chunks: {', '.join(nlp['noun_chunks']) or 'none'}\n"
        f"Named entities: {', '.join(e['text'] for e in nlp['entities']) or 'none'}"
    )


def generate_names(idea: str, nlp: dict, max_retries: int = 2) -> NamingResult:
    """Call the LLM, validate the JSON, and retry if it is invalid."""
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            response = _client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(idea, nlp)},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "naming_result",
                        "strict": True,
                        "schema": NamingResult.model_json_schema(),
                    },
                },
                temperature=0.8,
            )
            content = response.choices[0].message.content
            result = NamingResult.model_validate_json(content)

            # The task asks for 2-3 suggestions per category
            if len(result.legal_names) >= 2 and len(result.product_names) >= 2:
                result.legal_names = result.legal_names[:3]
                result.product_names = result.product_names[:3]
                return result
            last_error = "LLM returned fewer than 2 suggestions"
        except (ValidationError, json.JSONDecodeError) as e:
            last_error = e
    raise RuntimeError(f"LLM did not return valid JSON after retries: {last_error}")


# Quick manual test: python -m app.llm_client
if __name__ == "__main__":
    from app.nlp_extractor import extract_entities

    idea = "An AI-powered app that helps university students in Saudi Arabia find part-time jobs."
    nlp = extract_entities(idea)
    result = generate_names(idea, nlp)
    print(result.model_dump_json(indent=2))