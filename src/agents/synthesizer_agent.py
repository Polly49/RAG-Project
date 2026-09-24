"""
Synthesizer Agent Node (Answer Generation & Conflict Detection)
1. Grounds LLM response strictly on retrieved senior excerpts.
2. Identifies and surfaces contrasting/conflicting advice across seniors.
3. Generates clear markdown output with exact senior & file citations.
"""

from groq import Groq
import src.config as config
from src.agents.state import GraphState

def detect_conflicts(text: str) -> bool:
    """Checks if the answer identifies divergent senior advice."""
    signals = ["differ", "conflict", "disagree", "diverge", "on the other hand", "whereas", "varying advice"]
    return any(sig in text.lower() for sig in signals)

def synthesizer_node(state: GraphState) -> GraphState:
    query = state.get("user_query", "").strip()
    chunks = state.get("retrieved_chunks", [])

    if not chunks:
        state["final_answer"] = "No relevant placement notes found for this query. Please try rephrasing."
        state["has_conflict"] = False
        state["sources_used"] = []
        return state

    # Format context blocks with senior metadata
    context_blocks = []
    sources = set()
    for idx, c in enumerate(chunks, 1):
        senior = c.get("senior_id") or "Senior Contributor"
        source = c.get("source_name") or "Guide"
        sources.add(f"{senior} ({source})")
        context_blocks.append(f"[Source #{idx} | {senior} | {source}]\n{c.get('text', '')}")

    context_text = "\n\n---\n\n".join(context_blocks)

    system_prompt = """You are the Placement Prep Knowledge Assistant.
Answer the student's question using ONLY the provided Source Excerpts.

RULES:
1. STRICT GROUNDING: Do not hallucinate outside facts.
2. CONFLICT DETECTION: If seniors give differing advice (e.g. DSA vs Projects), explicitly highlight the difference:
   '⚠️ Note: Sources offer differing guidance: [Senior A] suggests X, while [Senior B] suggests Y.'
3. CITATIONS: Cite the senior/file name for specific tips or patterns.
4. STRUCTURE: Use clean markdown headers and bullet points."""

    user_prompt = f"STUDENT QUESTION:\n{query}\n\nRETRIEVED EXCERPTS:\n{context_text}\n\nSynthesize a clear response:"

    client = Groq(api_key=config.GROQ_API_KEY)
    try:
        response = client.chat.completions.create(
            model=config.DEFAULT_GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            max_tokens=1000
        )
        answer = response.choices[0].message.content.strip()
        state["final_answer"] = answer
        state["has_conflict"] = detect_conflicts(answer)
        state["sources_used"] = sorted(list(sources))
    except Exception as e:
        state["final_answer"] = f"Error generating answer: {e}"
        state["has_conflict"] = False
        state["sources_used"] = sorted(list(sources))

    return state
