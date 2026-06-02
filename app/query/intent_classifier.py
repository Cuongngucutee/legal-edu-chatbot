"""
LawEdu AI — Intent Classifier.
Replaces the QwenIntentClassifier (1.5B local model) with Pro LLM API calls.
Keeps the fast heuristic rules (loaded from Ontology) + uses 320B for ambiguous cases.
"""
import re
import json
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Valid intent types
VALID_INTENTS = {
    "LOOKUP", "SUMMARY", "LISTING", "STATISTICAL",
    "COMPARISON", "UNANSWERABLE", "GREETING", "THANKS",
}

# Load Ontology
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ONTOLOGY_PATH = os.path.join(BASE_DIR, "data", "domain_knowledge.json")

try:
    with open(ONTOLOGY_PATH, "r", encoding="utf-8") as f:
        _ontology = json.load(f).get("intent_classifier", {})
        ONTOLOGY_KEYWORDS = _ontology.get("keywords", {})
        ONTOLOGY_LOOKUP_CONTEXT = _ontology.get("lookup_context", [])
        ONTOLOGY_LEGAL_KEYWORDS = set(_ontology.get("legal_keywords", []))
except Exception as e:
    logger.error(f"Failed to load ontology {ONTOLOGY_PATH}: {e}")
    ONTOLOGY_KEYWORDS = {}
    ONTOLOGY_LOOKUP_CONTEXT = []
    ONTOLOGY_LEGAL_KEYWORDS = set()


def get_fast_intent(query: str) -> Optional[str]:
    """Fast check for greetings and thanks without LLM using Ontology."""
    q = query.lower().strip()
    greetings = ONTOLOGY_KEYWORDS.get("GREETING", [])
    thanks = ONTOLOGY_KEYWORDS.get("THANKS", [])
    if len(q) < 30 and any(q.startswith(g) or q == g for g in greetings):
        return "GREETING"
    if len(q) < 40 and any(kw in q for kw in thanks):
        return "THANKS"
    return None

def classify_intent(query: str, llm) -> str:
    """
    Classify query intent. Priority: fast heuristics (Ontology) → Pro LLM fallback.
    """
    fast_intent = get_fast_intent(query)
    if fast_intent:
        return fast_intent

    q = query.lower().strip()

    # ── Heuristic rules from Ontology ──
    if any(kw in q for kw in ONTOLOGY_KEYWORDS.get("SUMMARY", [])):
        return "SUMMARY"

    if any(kw in q for kw in ONTOLOGY_KEYWORDS.get("STATISTICAL", [])):
        # Context-aware: "bao nhiêu" in lookup context → LOOKUP, not STATISTICAL
        if any(kw in q for kw in ONTOLOGY_LOOKUP_CONTEXT):
            return "LOOKUP"
        if any(kw in q for kw in ONTOLOGY_KEYWORDS.get("UNANSWERABLE", [])):
            return "UNANSWERABLE"
        return "STATISTICAL"

    if any(kw in q for kw in ONTOLOGY_KEYWORDS.get("LISTING", [])):
        return "LISTING"

    if any(kw in q for kw in ONTOLOGY_KEYWORDS.get("COMPARISON", [])):
        return "COMPARISON"

    # ── P2: Legal education keyword heuristic ──
    match_count = sum(1 for kw in ONTOLOGY_LEGAL_KEYWORDS if kw in q)
    if match_count >= 2 and len(q) > 20:
        logger.info(f"🏷️  Intent heuristic: LOOKUP (matched {match_count} legal keywords)")
        return "LOOKUP"

    # ── Pro LLM classification for truly ambiguous cases (< 5%) ──
    from app.llm.prompts import INTENT_PROMPT
    try:
        result = llm.generate(INTENT_PROMPT.format(query=query), max_tokens=10)
        intent = result.strip().upper().split('\n')[0].strip()
        if intent in VALID_INTENTS:
            return intent
    except Exception as e:
        logger.warning(f"LLM intent classification failed: {e}")

    return "LOOKUP"  # Safe default
