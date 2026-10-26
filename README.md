# Startup Namer — NLP + LLM Naming Prototype

A smart onboarding prototype for an entrepreneurship mentoring platform.
The user enters a startup idea (Arabic or English, up to 300 characters); the app
extracts key concepts with **classical NLP**, then sends the text and the extracted
entities to an **LLM** that returns legal company names and product names as **structured JSON**.

## Setup & Run

**Requirements:** Python 3.13, a free [Groq API key](https://console.groq.com/keys).

```bash
git clone <this-repo-url>
cd startup_namer

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
python -m spacy download en_core_web_sm

copy .env.example .env         # Windows (use `cp` on macOS / Linux)
# then put your key in .env:  GROQ_API_KEY=gsk_...

uvicorn app.main:app
```
Open **http://127.0.0.1:8000**. The first run downloads the KeyBERT embedding model (~470 MB).

## Project Structure
```
app/
├── nlp_extractor.py   # classical NLP pipeline (no LLM)
├── llm_client.py      # Groq call + JSON enforcement
├── schemas.py         # Pydantic data models
└── main.py            # FastAPI routes
static/index.html      # frontend
```

## NLP Extraction Approach
The pipeline uses **no LLM**:

1. **Language detection** with `langdetect` (seeded for deterministic results).
2. **spaCy** (`en_core_web_sm`) extracts **named entities** (e.g. `Saudi Arabia → GPE`)
   and **noun chunks** (e.g. `university students`, `part-time jobs`) for English.
   Determiners and pronouns are removed from chunks.
3. **KeyBERT** with the multilingual model `paraphrase-multilingual-MiniLM-L12-v2`
   ranks phrases by **semantic similarity** to the whole idea, using **MMR**
   (`diversity=0.5`) to avoid near-duplicate keywords.

**Why this combination:** for English, spaCy's noun chunks are passed to KeyBERT as
**candidates**, so KeyBERT ranks linguistically valid phrases instead of generating
raw n-grams. This fixed noisy outputs such as `time jobs` (caused by stop-word removal
of `part`). The multilingual embedding model lets the same pipeline support Arabic.

**Known limitation:** spaCy has no official Arabic pipeline, so Arabic keywords come
from KeyBERT n-grams only and may occasionally cross phrase boundaries.
A possible improvement is using CAMeL Tools or Stanza for Arabic noun phrases.

## LLM Prompt & JSON Enforcement
- **Model:** `openai/gpt-oss-120b` via Groq (configurable with `GROQ_MODEL` in `.env`).
- **System prompt** defines the role (branding consultant) and strict rules:
  3 formal legal names (English ends with `LLC`, Arabic starts with `شركة`),
  3 catchy product names in the input language, one-sentence rationales.
- **User prompt** combines the **original idea** with the **extracted NLP entities**
  (language, keywords, noun chunks, named entities) to ground the suggestions.
- **JSON is enforced in two layers:**
  1. Groq **Structured Outputs** (`response_format: json_schema`, `strict: true`),
     with the schema generated from the Pydantic `NamingResult` model.
  2. **Pydantic validation** of the response, with up to 2 retries if it is invalid
     or contains fewer than 2 suggestions.

## API
`POST /api/generate` — body: `{"idea": "..."}` → response:
```json
{
  "nlp": { "language": "en", "keywords": [], "entities": [], "noun_chunks": [] },
  "suggestions": {
    "legal_names":   [{ "name": "...", "rationale": "..." }],
    "product_names": [{ "name": "...", "rationale": "..." }]
  }
}
```

## Security
The API key is read from `.env` (git-ignored). Only `.env.example` is committed.

## AI Tools Used
- **Claude (Anthropic):** generated the frontend (`static/index.html`) and the FastAPI
  routing boilerplate (`app/main.py`). Also used as a tutor and code reviewer while
  building and debugging the NLP and LLM modules.