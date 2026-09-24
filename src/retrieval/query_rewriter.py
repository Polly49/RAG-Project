"""
Multi-Query Rewriter (Query Expansion)

AI/ML Concept:
- A user query like 'sql tips' may fail to match documents about 'window functions' or 'queries'.
- Using an LLM to generate 3 alternative domain-specific query variations improves retrieval recall.
"""

import re
from groq import Groq
import src.config as config

_GROQ_CLIENT = None

def get_groq_client() -> Groq:
    global _GROQ_CLIENT
    if _GROQ_CLIENT is None:
        _GROQ_CLIENT = Groq(api_key=config.GROQ_API_KEY)
    return _GROQ_CLIENT

def rewrite_query(original_query: str, n: int = 3) -> list[str]:
    """Generates n alternative search queries using Groq LLM."""
    if not original_query or not original_query.strip():
        return [original_query]

    client = get_groq_client()
    prompt = f"""Generate {n} different specific placement search queries for: "{original_query}".
Rules: Output ONLY the {n} query variations, one per line. No numbers, no extra text."""

    try:
        response = client.chat.completions.create(
            model=config.DEFAULT_GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=150
        )
        lines = response.choices[0].message.content.strip().split("\n")
        rewrites = [re.sub(r"^[\d\.\-\*\•\s]+", "", l).strip() for l in lines if l.strip()]
        return [original_query] + rewrites[:n]
    except Exception:
        return [original_query]

if __name__ == "__main__":
    print(rewrite_query("sql interview prep"))
