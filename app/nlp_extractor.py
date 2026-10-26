"""
nlp_extractor.py
Classical NLP pipeline (no LLM):
language detection -> spaCy noun chunks & entities -> KeyBERT ranking.
"""
from langdetect import detect, DetectorFactory
from keybert import KeyBERT
import spacy

# Make language detection give the same result every run
DetectorFactory.seed = 0

# Load models ONCE (loading is slow, so we don't repeat it per request)
_kw_model = KeyBERT(model="paraphrase-multilingual-MiniLM-L12-v2")
_nlp_en = spacy.load("en_core_web_sm")


def detect_language(text: str) -> str:
    """Return 'ar' or 'en'. Falls back to 'en' if detection fails."""
    try:
        lang = detect(text)
    except Exception:
        return "en"
    return "ar" if lang == "ar" else "en"


def extract_keywords(text: str, lang: str, top_n: int = 6,
                     candidates: list[str] | None = None) -> list[str]:
    """Rank key phrases with KeyBERT.
    If candidates are given (English noun chunks), KeyBERT ranks them
    by semantic similarity instead of generating its own n-grams."""
    results = _kw_model.extract_keywords(
        text,
        candidates=candidates,
        keyphrase_ngram_range=(1, 3) if candidates else (1, 2),
        stop_words=None,
        use_mmr=True,
        diversity=0.5,
        top_n=top_n,
    )
    return [phrase for phrase, score in results]


def extract_english_structure(text: str) -> dict:
    """Named entities and noun chunks via spaCy (English only)."""
    doc = _nlp_en(text)
    entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
    noun_chunks = []
    for chunk in doc.noun_chunks:
        # Drop determiners/pronouns: "an ai-powered app" -> "ai-powered app"
        phrase = "".join(
            t.text_with_ws for t in chunk if t.pos_ not in ("DET", "PRON")
        ).strip().lower()
        if len(phrase) > 2 and phrase not in noun_chunks:
            noun_chunks.append(phrase)
    return {"entities": entities, "noun_chunks": noun_chunks}


def extract_entities(text: str) -> dict:
    """Main entry point: run the full NLP pipeline on the user's idea."""
    text = text.strip()
    lang = detect_language(text)
    result = {"language": lang, "keywords": [], "entities": [], "noun_chunks": []}

    if lang == "en":
        result.update(extract_english_structure(text))

    # Hyphens -> spaces so candidates match KeyBERT's tokenizer ("part-time" -> "part time")
    candidates = [c.replace("-", " ") for c in result["noun_chunks"]] or None
    result["keywords"] = extract_keywords(text, lang, candidates=candidates)
    return result


# Quick manual test: python -m app.nlp_extractor
if __name__ == "__main__":
    samples = [
        "An AI-powered app that helps university students in Saudi Arabia find part-time jobs.",
        "منصة إلكترونية تربط المعلمين الخصوصيين بأولياء الأمور لتعليم الأطفال القرآن الكريم عن بعد.",
    ]
    for s in samples:
        print(extract_entities(s))
        print("-" * 40)