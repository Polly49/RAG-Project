"""
Multi-Query Rewriter using LangChain Structured Outputs
Guarantees strictly formatted query variations using Pydantic schema.
"""

import os
import sys
from pathlib import Path
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.config as config

# Pydantic schema for structured output
class QueryVariations(BaseModel):
    """Structured variations of a search query for placement preparation."""
    variations: list[str] = Field(
        description="3 distinct search queries covering key concepts, technical synonyms, and interview patterns"
    )

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AI placement search assistant. Generate {n} distinct search query variations for the student's question to maximize relevant placement note retrieval."),
    ("user", "{query}")
])

def rewrite_query(original_query: str, n: int = 3) -> list[str]:
    """Generates schema-validated alternative search queries using LangChain structured output."""
    if not original_query or not original_query.strip():
        return [original_query]

    api_key = os.getenv("GROQ_API_KEY") or config.GROQ_API_KEY
    if not api_key:
        return [original_query]

    try:
        llm = ChatGroq(api_key=api_key, model=config.DEFAULT_GROQ_MODEL, temperature=0.3)
        structured_llm = llm.with_structured_output(QueryVariations)
        chain = prompt | structured_llm
        result: QueryVariations = chain.invoke({"query": original_query, "n": n})
        return [original_query] + result.variations[:n]
    except Exception:
        return [original_query]

if __name__ == "__main__":
    print(rewrite_query("sql interview prep"))
