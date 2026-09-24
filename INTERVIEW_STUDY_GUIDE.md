# 🎓 Placement Prep Knowledge Assistant — Comprehensive Interview Study Guide & Code Explainer

This master guide is tailored for **college placement interviews and technical RAG/AI system design rounds**. It is split into three essential parts:
1. **Fundamental Concepts to Study Before Diving Into the Code**
2. **File-by-File & Function-by-Function In-Depth Code Explanations**
3. **Top Interview Questions & Battle-Tested Model Answers**

---

# PART 1: Core Concepts to Master First

Before walking an interviewer through the code, make sure you know these **7 foundational concepts** cold.

---

### 1. Bi-Encoder (Dense Vectors) vs. Sparse Keyword Search (BM25)

```
Query: "InfoEdge ML interview questions"

Dense Search (Bi-Encoder):
  Embed(Query)  ──────┐ (Cosine Similarity)
  Embed(Chunk)  ──────┴──► Good for synonyms ("ML questions" ≈ "Machine Learning test")
                          Weak on exact tokens ("InfoEdge", "ROW_NUMBER", "LPA")

Sparse Search (BM25Okapi):
  Tokens(Query) ──────┐ (Term Frequency & Inverse Doc Frequency)
  Tokens(Chunk) ──────┴──► Perfect for exact tokens ("InfoEdge", "ROW_NUMBER")
                          Weak on synonyms ("DSA" vs "Data Structures and Algorithms")
```

- **Dense Embedding (Bi-Encoder):**
  - Model used: `all-MiniLM-L6-v2` (SentenceTransformer).
  - Converts text into a **384-dimensional dense floating-point vector**.
  - Query vector $\vec{q}$ and document vector $\vec{d}$ are compared using **Cosine Similarity**:
    $$\text{Cosine Similarity}(\vec{q}, \vec{d}) = \frac{\vec{q} \cdot \vec{d}}{\|\vec{q}\|_2 \|\vec{d}\|_2}$$
  - In our code, we normalize all vectors to unit L2 length (`faiss.normalize_L2`), so Inner Product (`IndexFlatIP`) equals exact Cosine Similarity.
- **Sparse Keyword Search (BM25Okapi):**
  - Uses term frequency ($TF$) and inverse document frequency ($IDF$) with document length normalization:
    $$\text{BM25 Score} = \sum_{t \in Q} IDF(t) \cdot \frac{TF(t, D) \cdot (k_1 + 1)}{TF(t, D) + k_1 \cdot (1 - b + b \cdot \frac{|D|}{\text{avgdl}})}$$
  - Matches exact words, acronyms, and company names.

---

### 2. Reciprocal Rank Fusion (RRF) — The Fusion Math

```
Dense Ranking:   [Doc A (Rank 1), Doc B (Rank 2), Doc C (Rank 3)]
BM25 Ranking:    [Doc C (Rank 1), Doc A (Rank 2), Doc D (Rank 3)]

RRF Score (k=60):
  Doc A = 1/(60 + 1) + 1/(60 + 2) = 0.01639 + 0.01612 = 0.03251  (1st Place)
  Doc C = 1/(60 + 3) + 1/(60 + 1) = 0.01587 + 0.01639 = 0.03226  (2nd Place)
```

- **Why RRF instead of averaging raw scores?**
  - Dense cosine similarity is bounded between $[0, 1]$.
  - BM25 score is unbounded (ranges between $[0, 50+]$ depending on term frequency).
  - Averaging raw scores is statistically invalid. **RRF relies strictly on relative rank positions**, making it scale-invariant and robust across different retrieval algorithms.

---

### 3. Cross-Encoder Reranking (`ms-marco-MiniLM-L-6-v2`)

```
Bi-Encoder (FAISS):
  Query  ──► [Transformer A] ──► Vector Q ──┐ (Fast Dot Product O(N))
  Doc    ──► [Transformer B] ──► Vector D ──┘

Cross-Encoder:
  [Query + " [SEP] " + Doc] ──► [Full Self-Attention Transformer] ──► Score (Slower O(K), Highly Accurate)
```

- In a Bi-Encoder, the query and document never "see" each other during token encoding.
- In a Cross-Encoder, all query tokens attend to all document tokens across all self-attention layers simultaneously.
- **System Design Trade-off:** Running a cross-encoder over 100,000 chunks is too slow ($O(N)$). Running it on the **top-20 hybrid candidates** takes only ~20ms and provides maximal precision.

---

### 4. Query Expansion / Multi-Query Rewriting

- Junior queries are often short, informal, or ambiguous (e.g. *"sql tips"* or *"deloitte format"*).
- We use a fast Groq LLM (`openai/gpt-oss-20b` or `qwen/qwen3.6-27b`) to generate 3 alternative domain-specific queries.
- We retrieve across all 4 queries, merge, and deduplicate candidates before reranking.

---

### 5. Multi-Agent System Architecture (LangGraph)

Why not just one giant prompt?
1. **Separation of Concerns:** Intent classification (Router), Information Gathering (Retrieval), and Synthesis (Synthesizer) can be tested, monitored, and scaled independently.
2. **Defensive Fallbacks:** If retrieval returns 0 candidates, the Synthesizer doesn't hallucinate—it yields an honest fallback.
3. **State Management:** All nodes share a typed state (`GraphState`), ensuring predictable data flow.

---

### 6. Conflict & Divergence Detection (The Core Differentiator)

- In campus placements, **seniors often disagree**:
  - *Senior A (SDE Focus):* "Grind Striver's SDE Sheet and LeetCode Mediums first."
  - *Senior B (Analytics Focus):* "Prioritize SQL window functions, guesstimates, and case frameworks."
  - *Company Roles:* Deloitte Analyst test has 65 MCQs; Deloitte Senior Executive test has ML debugging.
- **Naive RAG fails:** It mixes both chunks into one blended answer, confusing the student.
- **Our Synthesizer Agent:** Instructed to actively scan for divergent advice across sources and surface it clearly under warning callouts with explicit citations.

---

# PART 2: File-by-File & Line-by-Line Code Breakdown

---

## 1. `src/config.py` — Central Configuration Hub

```python
# src/config.py
import os
from dotenv import load_dotenv

CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CONFIG_DIR)
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
```

### Line-by-Line Explanation:
- **`CONFIG_DIR` & `PROJECT_ROOT`:** Computes absolute file paths dynamically so scripts run cleanly from any working directory on Windows, Linux, or macOS.
- **`load_dotenv(...)`:** Reads `.env` to securely load API keys and hyper-parameters without hardcoding secrets.
- **`TOPICS`:** A fixed list of permitted categories (`dsa`, `sql`, `aptitude`, `resume`, `hr_interview`, `system_design`, `core_cs`, `genai_ds`, `general`).
- **`CHUNK_SIZE = 400`, `CHUNK_OVERLAP = 50`:** Defines our semantic chunking window size (~400 words with 50 words sliding overlap).
- **`TOP_K_RETRIEVE = 20`, `TOP_K_RERANK = 5`:** Hybrid search pulls 20 candidates; Cross-Encoder reranker trims to the top 5.

---

## 2. `src/ingestion/ingest_github.py` — GitHub Repository Ingestion

```python
# src/ingestion/ingest_github.py
def download_repo_archive(owner, repo_name, dest_dir):
    # Downloads repository archive directly as ZIP from GitHub
```

### Why this was built:
1. Many Windows systems don't have `git.exe` in `$PATH`.
2. `download_repo_archive` fetches the repository zip via HTTP (`https://github.com/{owner}/{repo_name}/archive/refs/heads/main.zip`) and extracts it with Python's built-in `zipfile` module.
3. `guess_topic(filepath, default_topic)`: Inspects folder and file names (e.g. `02_Aptitude` $\rightarrow$ `aptitude`, `04_Data_Science_ML` $\rightarrow$ `genai_ds`).
4. `guess_company(filename)`: Detects if the markdown file is a company interview report (`Deloitte_India.md` $\rightarrow$ `Deloitte`, `InfoEdge.md` $\rightarrow$ `InfoEdge`).
5. Outputs normalized JSON Lines records to `data/processed/github_chunks_raw.jsonl`.

---

## 3. `src/ingestion/ingest_text.py` — Senior Text Notes Ingestion

```python
# src/ingestion/ingest_text.py
def parse_filename(filename):
    # Format: <senior_id>__<topic>__<description>.txt
    parts = base_name.split("__")
```

### Why this was built:
1. Allows students/seniors to drop text notes into `data/raw/text_notes/` without maintaining a separate database.
2. Uses the double-underscore convention `sameer__sql__window_functions.txt` to parse metadata (`senior_id="sameer"`, `topic="sql"`).
3. If the file doesn't follow the convention, it safely defaults to `senior_id="unknown"`, `topic="general"`.
4. Outputs to `data/processed/text_chunks_raw.jsonl`.

---

## 4. `src/ingestion/scrape_web.py` — Google Workspace & Web Scraper

```python
# src/ingestion/scrape_web.py
def handle_google_doc(url, default_topic, company):
    # Rewrites doc URL to: https://docs.google.com/document/d/<ID>/export?format=txt

def handle_google_sheet(url, default_topic, company):
    # Rewrites sheet URL to: https://docs.google.com/spreadsheets/d/<ID>/export?format=csv&gid=<GID>
```

### Why this was built:
1. **Google Docs Trick:** Standard `requests.get()` on Google Docs returns an empty JavaScript page. Converting the URL to `/export?format=txt` downloads the complete unformatted text directly (extracted **88,700+ characters** from our placement doc).
2. **Google Sheets Interview Chronicler:** Converting Google Sheets to `/export?format=csv` allows row-by-row iteration. Each row (Company, Role, Test Date, CTC, Test Pattern, Interview Rounds) is converted into structured natural language:
   ```text
   Company: Deloitte
   Role: Analyst
   Test Pattern: 65 MCQs in 65 minutes covering quant, verbal, DILR
   Interview Rounds: 20-min panel on DL & transformer over-fitting
   ```
3. **HTML Article Scraper:** Uses `BeautifulSoup4` to decompose `<script>`, `<style>`, `<nav>`, `<header>`, `<footer>` tags, extracting clean interview puzzle articles from GeeksforGeeks and resource blogs.
4. Outputs to `data/processed/web_chunks_raw.jsonl`.

---

## 5. `src/ingestion/ingest_pdf.py` — PDF Casebook & Note Parser

```python
# src/ingestion/ingest_pdf.py
import pdfplumber

def extract_pdf_text(filepath):
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
```

### Why this was built:
1. Placement repositories frequently store casebooks, interview frameworks, and cheat sheets as PDFs.
2. Uses `pdfplumber` for superior table and multi-column layout extraction compared to basic PyPDF2.
3. Automatically skips Git LFS pointer files or scanned image PDFs gracefully without crashing.

---

## 6. `src/processing/merge_and_tag.py` & `src/processing/chunker.py`

```python
# src/processing/chunker.py
def split_text_into_chunks(text, max_words=400, overlap=50):
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
```

### Why this was built:
1. **`merge_and_tag.py`:** Reads all four raw JSONL files (`github`, `text`, `web`, `pdf`), removes duplicate entries using a content-hash set, and saves `merged_raw.jsonl`.
2. **`chunker.py` (Paragraph-Aware):**
   - Naive token splitters cut sentences in half mid-thought.
   - Our chunker groups text by **paragraph boundaries (`\n\n`)**, accumulating paragraphs up to ~400 words.
   - If a single paragraph is too large, it applies a 50-word sliding window.
   - Generates a unique `chunk_id` for every piece while inheriting parent metadata (`senior_id`, `company`, `topic`, `source_name`).
   - Produces the master retrieval corpus: `data/processed/chunks.jsonl` (136 chunks).

---

## 7. `src/retrieval/build_index.py` — Dual Index Generation

```python
# src/retrieval/build_index.py
# 1. Dense Index (FAISS)
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embed_model.encode(texts, convert_to_numpy=True)
faiss.normalize_L2(embeddings)
faiss_index = faiss.IndexFlatIP(384)
faiss_index.add(embeddings)

# 2. Sparse Index (BM25)
tokenized_corpus = [re.findall(r"\w+", t.lower()) for t in texts]
bm25 = BM25Okapi(tokenized_corpus)
```

### Why this was built:
1. **Dense Vector Index:** Computes 384-dimensional embeddings with `all-MiniLM-L6-v2`. L2-normalizes vectors so `faiss.IndexFlatIP` computes exact cosine similarity via fast inner product matrix multiplication.
2. **Sparse Keyword Index:** Builds a tokenized inverted index using `rank_bm25.BM25Okapi`.
3. **Metadata Mapping:** Writes `indexes/chunk_metadata.jsonl` matching the internal array indexing of both FAISS and BM25.

---

## 8. `src/retrieval/hybrid_retriever.py` — Reciprocal Rank Fusion Engine

```python
# src/retrieval/hybrid_retriever.py
def reciprocal_rank_fusion(dense_results, sparse_results, k=60):
    rrf_scores = {}
    for rank, (doc_idx, _) in enumerate(dense_results, start=1):
        rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0) + 1.0 / (k + rank)
    for rank, (doc_idx, _) in enumerate(sparse_results, start=1):
        rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0) + 1.0 / (k + rank)
    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
```

### Why this was built:
1. `load_indexes()`: Caches the FAISS index, BM25 pickle, and SentenceTransformer model in memory once.
2. `dense_search(query, k=30)`: Normalizes query vector and searches FAISS.
3. `sparse_search(query, k=30)`: Tokenizes query and ranks documents by BM25 score.
4. `reciprocal_rank_fusion(...)`: Implements RRF with constant $k=60$ to combine dense and sparse rankings without score scale mismatch.
5. `hybrid_search(query, top_k=20)`: Returns top-20 fused candidate chunks with full metadata.

---

## 9. `src/retrieval/query_rewriter.py` — Multi-Query Expansion

```python
# src/retrieval/query_rewriter.py
def rewrite_query(original_query, n=3, model=config.DEFAULT_GROQ_MODEL):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        temperature=0.3
    )
```

### Why this was built:
1. Prompts the Groq LLM to generate 3 alternative domain-specific queries.
2. Cleans bullet numbers and whitespace.
3. Returns `[original_query, rewrite_1, rewrite_2, rewrite_3]`.
4. If the LLM call times out or fails, it cleanly falls back to `[original_query]` so retrieval never crashes.

---

## 10. `src/retrieval/reranker.py` — Cross-Encoder Precision Reranker

```python
# src/retrieval/reranker.py
from sentence_transformers import CrossEncoder

def rerank(query, candidate_chunks, top_k=5):
    model = get_reranker_model("cross-encoder/ms-marco-MiniLM-L-6-v2")
    pairs = [[query, chunk.get("text", "")] for chunk in candidate_chunks]
    scores = model.predict(pairs)
```

### Why this was built:
1. Bi-encoders retrieve a broad pool of 20 candidates.
2. The cross-encoder evaluates full joint self-attention on each `(query, document)` pair.
3. Attaches `rerank_score` to each candidate and sorts descending.
4. Returns the top 5 highest-precision chunks.

---

## 11. `src/retrieval/pipeline.py` — Full Retrieval Pipeline

```python
# src/retrieval/pipeline.py
def retrieve_and_rerank(user_query, topic_filters=None, company_filter=None, top_k_final=5):
    # 1. Query Expansion (3 variations)
    # 2. Hybrid Search for each query variant
    # 3. Deduplication by chunk_id
    # 4. Topic & Company filtering
    # 5. Cross-Encoder Reranking to top-5
```

### Why this was built:
- Orchestrates the full retrieval flow into a single callable function.
- Tracks execution timings (`query_rewrite_ms`, `hybrid_search_ms`, `reranking_ms`, `total_pipeline_ms`) for performance monitoring.

---

## 12. `src/agents/state.py` — LangGraph Shared State

```python
# src/agents/state.py
class GraphState(TypedDict):
    user_query: str
    topic_filters: Optional[List[str]]
    company_filter: Optional[str]
    needs_synthesis: bool
    retrieved_chunks: List[Dict[str, Any]]
    final_answer: str
    has_conflict: bool
    sources_used: List[str]
```

### Why this was built:
- Defines the data schema that flows through each node in the LangGraph multi-agent pipeline.
- Ensures type safety and strict separation of state mutations across nodes.

---

## 13. `src/agents/router_agent.py` — Query Classifier Node

```python
# src/agents/router_agent.py
def router_node(state: GraphState) -> GraphState:
    # Classifies user query as 'SIMPLE' vs 'SYNTHESIS'
```

### Why this was built:
- Evaluates if the query is a simple factual question (e.g. *"What is the CTC for BlackRock?"*) or a strategic synthesis question (e.g. *"Should I prepare DSA or ML first?"*).
- Sets `state["needs_synthesis"] = True/False`.

---

## 14. `src/agents/retrieval_agent.py` — Tool Node

```python
# src/agents/retrieval_agent.py
def retrieval_node(state: GraphState) -> GraphState:
    results = retrieve_and_rerank(...)
    state["retrieved_chunks"] = results.get("reranked_chunks", [])
```

### Why this was built:
- Wraps the retrieval pipeline as a modular LangGraph node.
- If retrieval yields zero results, it passes an empty list so the Synthesizer can provide a clean fallback.

---

## 15. `src/agents/synthesizer_agent.py` — Conflict-Aware Generation Node

```python
# src/agents/synthesizer_agent.py
def synthesizer_node(state: GraphState) -> GraphState:
    # 1. Formats chunks with Senior ID, Company, Topic, Source tags
    # 2. Strict grounding instructions (no hallucination)
    # 3. Explicit conflict detection instructions
    # 4. Extracts cited sources and sets has_conflict flag
```

### Why this was built:
- Prompts Groq LLM with retrieved excerpts.
- **Explicit Conflict Detection Instruction:** If two seniors give contrasting advice on prep or test patterns, it surfaces both viewpoints under `⚠️ Note: Sources offer differing guidance: [Senior A] suggests X, while [Senior B] suggests Y.`
- Enforces in-line source citations (`[Senior Name - Source File]`).
- Detects conflict signal phrases to set `state["has_conflict"] = True`.

---

## 16. `src/agents/graph.py` — LangGraph Multi-Agent Orchestrator

```python
# src/agents/graph.py
workflow = StateGraph(GraphState)
workflow.add_node("router", router_node)
workflow.add_node("retrieval", retrieval_node)
workflow.add_node("synthesizer", synthesizer_node)

workflow.add_edge(START, "router")
workflow.add_edge("router", "retrieval")
workflow.add_edge("retrieval", "synthesizer")
workflow.add_edge("synthesizer", END)
```

### Why this was built:
- Compiles the linear multi-agent workflow into an executable graph.
- Exposes `run_query(user_query, topic_filters, company_filter)` to the Streamlit UI.

---

## 17. `src/app.py` — Streamlit Web Application

```python
# src/app.py
st.set_page_config(page_title="Placement Prep Knowledge Assistant", layout="wide")
```

### Why this was built:
1. **Sidebar:** Displays real-time corpus stats (Total Chunks, Verified Seniors, Company Logs), Topic multiselect filter, and Company selectbox filter.
2. **Main Interface:** Search box with one-click sample query buttons.
3. **Conflict Banner:** Dynamically renders an alert banner if `has_conflict` is True.
4. **Source Accordion:** Expandable section showing exact retrieved chunk snippets, contributor names, and rerank scores.

---

## 18. `src/eval/run_eval.py` — Retrieval Evaluation Benchmark

```python
# src/eval/run_eval.py
# Compares:
# 1. Naive (Dense FAISS alone)
# 2. Hybrid (BM25 + FAISS via RRF)
# 3. Multi-Query Hybrid + Cross-Encoder Reranker
```

### Why this was built:
- Runs 15 realistic placement prep queries across all 3 retrieval configurations.
- Calculates topic-match accuracy @ 5.
- Demonstrates that **Hybrid + Cross-Encoder Reranking** outperforms naive dense retrieval by capturing both exact keyword matches and semantic meaning.

---

# PART 3: Top Interview Questions & How to Answer Them

### Q1: "Can you give a 1-minute elevator pitch of this project?"
> *"I built the **Placement Prep Knowledge Assistant**, a multi-agent RAG system designed to help college students prepare for campus placements. Placement guidance is notoriously messy—scattered across senior GitHub repos, personal drive notes, CTC spreadsheets, and interview blogs—and different seniors often give conflicting advice. 
> To solve this, I built a hybrid retrieval pipeline combining **FAISS dense embeddings** with **BM25 keyword search** fused via **Reciprocal Rank Fusion (RRF)**, followed by a **Cross-Encoder reranker** for precision. The system is orchestrated using a 3-agent **LangGraph** pipeline (Router $\rightarrow$ Retrieval $\rightarrow$ Conflict Synthesizer) where the Synthesizer explicitly surfaces conflicting senior advice and cites verified sources instead of silently blending disparate viewpoints."*

---

### Q2: "Why did you use Hybrid Search (BM25 + Dense) instead of just Vector Search?"
> *"In placement preparation, queries often contain exact proper nouns, company names, or programming syntax—such as 'InfoEdge', 'Deloitte Analyst', 'ROW_NUMBER()', or 'B-Tree'. Dense embedding models (like all-MiniLM-L6-v2) excel at semantic concepts (mapping 'algorithm prep' to 'DSA roadmap'), but frequently produce false positives or miss specific company acronyms. BM25 provides exact token matching. Fusing them using Reciprocal Rank Fusion ensures we achieve both high semantic recall and exact keyword precision."*

---

### Q3: "What is Reciprocal Rank Fusion (RRF) and why not just average the scores?"
> *"Cosine similarity is bounded between 0 and 1, whereas BM25 produces unbounded scores from 0 to 50+ depending on term frequencies. Averaging raw scores from two completely different distributions is statistically invalid. RRF solves this by operating purely on the rank order:
> $$\text{RRF\_Score}(d) = \sum_{m} \frac{1}{k + \text{rank}_m(d)}$$
> where $k=60$. It awards higher scores to documents that appear near the top of either or both retrieval lists without requiring score normalization."*

---

### Q4: "Why use a Cross-Encoder for reranking?"
> *"Bi-encoders (used for FAISS) embed the query and document independently to enable fast vector lookups in $O(N)$ time. However, because query and document tokens are not cross-attended, subtle relationships like negations or precise constraints can be missed. A Cross-Encoder performs full self-attention over the combined query and document tokens simultaneously, yielding much higher precision. To balance latency and accuracy, we use Bi-Encoder hybrid search to retrieve the top 20 candidates, and then run the Cross-Encoder only on those 20 candidates to select the final top 5."*

---

### Q5: "How does your system handle conflicting information between seniors?"
> *"A major flaw of conventional RAG is silent averaging—if Senior A recommends focusing strictly on DSA while Senior B recommends SQL and project portfolios, standard models merge the two into a confusing answer. In our LangGraph pipeline, the Synthesizer Agent is explicitly prompted to identify contrasting advice. When divergent guidance is detected, the agent structures the answer comparatively, surfaces a conflict alert banner, and directly cites each senior's specific file."*

---

### Q6: "Why LangGraph instead of a single LLM prompt?"
> *"LangGraph allows us to build a modular state machine with clear separation of responsibilities:
> 1. The **Router Agent** determines if the query is a simple factual lookup or needs multi-source strategic synthesis.
> 2. The **Retrieval Tool** runs multi-query expansion, hybrid search, filtering, and cross-encoder reranking.
> 3. The **Synthesizer Agent** evaluates context, generates citations, and checks for conflicts.
> This makes each stage independently testable, reduces prompt complexity, and prevents hallucinations when no matching data is found."*
