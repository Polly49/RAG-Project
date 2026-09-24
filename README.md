# Placement Prep Knowledge Assistant

A multi-agent RAG system I built to make sense of all the scattered placement prep material that floats around during campus season — senior GitHub repos, interview logs, Google Docs, PDFs — all searchable from one place.

The part that took the most thought: seniors often give completely different advice for the same company or role. A naive RAG setup just blends everything together and loses that signal. So I built explicit conflict detection — the system actually flags when retrieved sources disagree and shows you both sides with citations.

---

## Why I built this

Every placement season I watched people (including me) spend hours piecing together advice from 10 different places. One senior says focus on Striver SDE sheet, another says LeetCode 75 with SQL is enough. One says Deloitte asks puzzles in round 1, another says it went straight to tech. None of that context survives a basic search.

I wanted a system that:
- Ingests all of that material automatically (GitHub repos, notes, Sheets)
- Lets you filter by company or topic
- Doesn't silently average out conflicting viewpoints

---

## Architecture

```
Data Sources
  Senior GitHub repos  →  ZIP download, extract .md / .txt
  Google Docs/Sheets   →  Export API
  Web / GfG puzzles    →  BeautifulSoup scraper
  PDFs                 →  pdfplumber
         |
         ↓
  Semantic chunker (~400 words, 50-word overlap)
         |
         ↓
  Hybrid index
    Dense  → FAISS (all-MiniLM-L6-v2, 384-dim)
    Sparse → BM25Okapi
         |
         ↓
  LangGraph pipeline
  Query → Router → Retrieval → Synthesizer → Answer
```

**Router** classifies the query as simple lookup vs synthesis. Synthesis queries need all conflicting chunks surfaced; simple ones just need the best match.

**Retrieval** does three things: expands the query into 3 rewrites using the LLM, runs hybrid BM25 + FAISS on all of them, merges with RRF (handles the score scale mismatch between BM25 and cosine similarity), then reranks top-20 to top-5 with a cross-encoder.

**Synthesizer** builds the final answer. If retrieved chunks have contradictory signals, it flags the conflict and shows both sides with source citations.

---

## Stack

| What | Tool |
|---|---|
| Agent orchestration | LangGraph |
| LLM | Groq (gpt-oss-20b / qwen3.6-27b) |
| Dense embeddings | sentence-transformers all-MiniLM-L6-v2 + FAISS |
| Sparse search | BM25Okapi |
| Rank fusion | Reciprocal Rank Fusion (RRF) |
| Reranking | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Scraping | BeautifulSoup4, pdfplumber |
| UI | Streamlit |

---

## Setup

```bash
git clone https://github.com/Polly49/RAG-Project.git
cd RAG-Project

python -m venv venv
.\venv\Scripts\activate       # Windows
# source venv/bin/activate   # Linux/macOS

pip install -r requirements.txt
```

Create a `.env` file:

```ini
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-20b
CHUNK_SIZE=400
CHUNK_OVERLAP=50
TOP_K_RETRIEVE=20
TOP_K_RERANK=5
```

Run ingestion and build the index:

```bash
python -m src.ingestion.ingest_github
python -m src.ingestion.ingest_text
python -m src.ingestion.scrape_web
python -m src.ingestion.ingest_pdf

python -m src.processing.merge_and_tag
python -m src.processing.chunker
python -m src.retrieval.build_index
```

Launch the UI:

```bash
streamlit run src/app.py
```

---

## Retrieval benchmark

| Strategy | Topic Relevance @ 5 |
|---|---|
| Naive dense (FAISS only) | 53.3% |
| Hybrid BM25 + FAISS | 64.0% |
| Multi-query hybrid + cross-encoder reranker | **82.7%** |

The gap between naive dense and the full pipeline is mostly BM25 recovering exact company names and terms that embeddings miss, and the cross-encoder fixing precision loss from the wide initial retrieval.

Full breakdown in `src/eval/results.md`.

---

## Design decisions

**BM25 alongside FAISS** — Dense embeddings are bad at exact matches. Company names like "InfoEdge" or SQL syntax get diluted. BM25 handles these precisely; running both and fusing gets the best of each.

**RRF for fusion** — BM25 scores can be 0-50+, cosine similarity is 0-1. You can't average those numbers. RRF fuses by rank position, not raw score, so it's scale-invariant.

**Cross-encoder reranking** — Bi-encoders encode query and document separately (fast, but coarse). A cross-encoder runs self-attention over the query-document pair together, which is much more accurate but slow. Retrieve top-20 fast with hybrid search, rerank to top-5 with cross-encoder.

**LangGraph for the pipeline** — Router, retrieval, and synthesizer as separate nodes means each is independently testable. Easier to swap out one without breaking the others.

---

## Limitations

- Conflict detection is heuristic (keyword signals + prompt). A fine-tuned classifier would be more reliable.
- JS-heavy pages like private Notion databases need explicit markdown/PDF exports — BeautifulSoup can't execute JS.
- Index rebuilding is full-batch. Incremental updates via Qdrant or Pinecone would be the natural next step.
