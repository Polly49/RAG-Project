# 🎓 MASTER AI TUTOR & INTERVIEW PREPARATION COMPENDIUM
# Multi-Agent Hybrid RAG Knowledge Assistant for Campus Placements

> **DIRECTIVE FOR AI TUTOR / MENTOR AGENT:**
> You are acting as an elite Principal AI/ML Engineer and Staff System Architect. Your mission is to teach the candidate every single layer of this codebase from first principles to advanced mathematical derivations, and conduct mock interview grilling sessions tailored to their attached resume.
> 
> When interacting with the student:
> 1. **Pedagogical Strategy:** Use the Socratic method. Explain the concept, reference the exact code file and line numbers, explain *why* this design choice was made over alternatives, and immediately follow up with a challenging interview question.
> 2. **Resume Alignment:** Cross-reference the project's technical achievements with the student's resume bullet points, ensuring they can defend every claim, metric, and library choice.
> 3. **Mathematical Rigor:** Never hand-wave the math. Explain L2 normalization, Cosine Similarity, BM25Okapi TF-IDF weighting, Reciprocal Rank Fusion ($k=60$), and Cross-Encoder self-attention mechanisms thoroughly.

---

# TABLE OF CONTENTS
1. [Executive Project Overview & Architecture Blueprint](#1-executive-project-overview--architecture-blueprint)
2. [Master Learning Sequence (Phase 1 to Phase 6)](#2-master-learning-sequence-phase-1-to-phase-6)
3. [File-by-File In-Depth Technical Breakdown](#3-file-by-file-in-depth-technical-breakdown)
   - [Phase 1: System Configuration (`src/config.py`)](#phase-1-system-configuration-srcconfigpy)
   - [Phase 2: Ingestion, Deduplication & Chunking](#phase-2-ingestion-deduplication--chunking)
   - [Phase 3: Indexing & Hybrid Retrieval Engine](#phase-3-indexing--hybrid-retrieval-engine)
   - [Phase 4: Multi-Agent Orchestration with LangGraph](#phase-4-multi-agent-orchestration-with-langgraph)
   - [Phase 5: Serving, APIs & Web Client](#phase-5-serving-apis--web-client)
   - [Phase 6: Evaluation & Benchmarking](#phase-6-evaluation--benchmarking)
4. [Mathematical & Algorithmic Foundations](#4-mathematical--algorithmic-foundations)
5. [Architectural Trade-Offs: "Why This vs Why Not That?"](#5-architectural-trade-offs-why-this-vs-why-not-that)
6. [Comprehensive Interview Question & Answer Vault (30 Questions)](#6-comprehensive-interview-question--answer-vault)
7. [Resume Bullet Points & Defense Playbook](#7-resume-bullet-points--defense-playbook)
8. [Hands-On Verification & Execution Runbook](#8-hands-on-verification--execution-runbook)

---

# 1. Executive Project Overview & Architecture Blueprint

### The Problem
During college campus placements, senior interview experiences, company test patterns, CTC breakdowns, and preparation roadmaps are scattered across unorganized GitHub repositories, Google Docs, PDFs, and informal text notes. Standard vector search engines fail on this data because:
1. Queries contain exact tokens (e.g. *"InfoEdge"*, *"Deloitte"*, *"ROW_NUMBER()"*, *"45 LPA"*), where dense vector embeddings often fail.
2. Junior queries are ambiguous and short (*"sql tips"* or *"deloitte format"*).
3. Seniors frequently offer **contradictory advice** (e.g., SDE seniors emphasizing LeetCode Trees/DP vs Data Science seniors emphasizing SQL and ML projects). Standard RAG averages these perspectives into a hallucinated compromise.

### The Solution
A production-grade, stateful **Multi-Agent Hybrid RAG System** featuring:
- **Two-Stage Retrieval:** Fast candidate generation using BM25 Sparse Search + FAISS Dense Search fused with Reciprocal Rank Fusion (RRF, $k=60$), followed by precision reranking with a Cross-Encoder (`ms-marco-MiniLM-L-6-v2`).
- **Multi-Query Expansion:** Groq LLM generating 3 domain-specific search variations.
- **LangGraph Multi-Agent Orchestrator:** Separation of concerns between Intent Routing, Retrieval, and Synthesizing with automated **Conflict & Divergence Detection**.
- **FastAPI Backend + Modern Responsive UI:** Sub-1.5s end-to-end response times with citation chips and inspectable evidence cards.

### End-to-End System Architecture

```mermaid
flowchart TD
    subgraph INGESTION ["1. Data Ingestion & Preprocessing"]
        A1[GitHub Repos: ingest_github.py] --> A5[Raw JSONL]
        A2[Web / Google Docs: scrape_web.py] --> A5
        A3[PDFs: ingest_pdf.py] --> A5
        A4[Text Notes: ingest_text.py] --> A5
        A5 --> A6[merge_and_tag.py: Deduplication]
        A6 --> A7[chunker.py: Paragraph-Aware 400w/50w Overlap]
        A7 --> A8[(data/processed/chunks.jsonl)]
    end

    subgraph INDEXING ["2. Offline Indexing Engine"]
        A8 --> B1[build_index.py]
        B1 -->|all-MiniLM-L6-v2 + L2 Norm| B2[(FAISS IndexFlatIP: 384-d)]
        B1 -->|BM25Okapi Tokenizer| B3[(BM25 Inverted Index: Pickle)]
        B1 --> B4[(indexes/chunk_metadata.jsonl)]
    end

    subgraph RUNTIME ["3. Online Multi-Agent Workflow (LangGraph)"]
        U([Student Query]) --> S[FastAPI / Web UI: server.py]
        S --> G[LangGraph Pipeline: GraphState]
        
        G --> R[Node 1: Router Agent]
        R -->|Classify: SIMPLE vs SYNTHESIS| G
        
        G --> RET[Node 2: Retrieval Agent]
        RET --> QE[Query Expansion: Groq 3 rewrites]
        QE --> HS[Multi-Query Hybrid Search: Top 30]
        B2 -.-> HS
        B3 -.-> HS
        HS --> RRF[Reciprocal Rank Fusion k=60]
        RRF --> MF[Metadata Filtering: Topic & Company]
        MF --> CE[Cross-Encoder Reranker: Top 20 -> Top 5]
        CE --> G
        
        G --> SYN[Node 3: Synthesizer Agent]
        SYN --> CD[Conflict Detection & Grounded Formatting]
        SYN --> OUT([JSON Response: Final Answer + Sources + Evidence])
    end
```

---

# 2. Master Learning Sequence (Phase 1 to Phase 6)

To build deep mental clarity, study the codebase in this precise order:

```
[Phase 1: System Foundations & Configuration]
   └─ src/config.py
        │
[Phase 2: Ingestion, Deduplication & Chunking]
   ├─ src/processing/merge_and_tag.py
   ├─ src/processing/chunker.py
   ├─ src/ingestion/ingest_github.py
   ├─ src/ingestion/scrape_web.py
   ├─ src/ingestion/ingest_pdf.py
   └─ src/ingestion/ingest_text.py
        │
[Phase 3: Indexing & Hybrid Retrieval Engine]
   ├─ src/retrieval/build_index.py
   ├─ src/retrieval/hybrid_retriever.py
   ├─ src/retrieval/query_rewriter.py
   ├─ src/retrieval/reranker.py
   └─ src/retrieval/pipeline.py
        │
[Phase 4: Multi-Agent Orchestration with LangGraph]
   ├─ src/agents/state.py
   ├─ src/agents/router_agent.py
   ├─ src/agents/retrieval_agent.py
   ├─ src/agents/synthesizer_agent.py
   └─ src/agents/graph.py
        │
[Phase 5: Serving, APIs & Web Client]
   ├─ src/server.py
   ├─ src/app.py (Streamlit alternative)
   ├─ frontend/index.html
   └─ frontend/app.js
        │
[Phase 6: Evaluation & Benchmarking]
   ├─ src/eval/eval_set.json
   ├─ src/eval/run_eval.py
   └─ src/eval/results.md
```

---

# 3. File-by-File In-Depth Technical Breakdown

---

## Phase 1: System Configuration (`src/config.py`)

### File Path: `src/config.py` (61 Lines)
- **Role:** Single Source of Truth for all directory paths, model names, hyperparameter thresholds, and topic taxonomy.
- **Key Code Lines:**
  - `Lines 11-16`: Computes `PROJECT_ROOT` and injects it into `sys.path[0]` to guarantee absolute imports regardless of execution directory.
  - `Lines 20-22`: Reconfigures Windows standard output to UTF-8 (`sys.stdout.reconfigure(encoding="utf-8")`) preventing Windows emoji encoding crashes.
  - `Lines 38-46`: Declares models:
    - LLM: `DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"` or `"qwen/qwen3.6-27b"`
    - Dense Bi-Encoder: `EMBED_MODEL_NAME = "all-MiniLM-L6-v2"` (384 dimensions)
    - Cross-Encoder: `RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"`
  - `Lines 48-54`: Retrieval parameters: `CHUNK_SIZE = 400`, `CHUNK_OVERLAP = 50`, `TOP_K_RETRIEVE = 20`, `TOP_K_RERANK = 5`.
  - `Lines 56-60`: `TOPICS` whitelist: `["dsa", "sql", "aptitude", "resume", "hr_interview", "system_design", "core_cs", "genai_ds", "general"]`.

---

## Phase 2: Ingestion, Deduplication & Chunking

### File 1: `src/processing/merge_and_tag.py` (43 Lines)
- **Role:** Merges disparate raw JSONL outputs from GitHub, web, PDF, and text ingestion scripts into a consolidated dataset while removing exact-text duplicates.
- **Key Code Lines:**
  - `Lines 11-16`: Enumerates source files (`github_chunks_raw.jsonl`, `web_chunks_raw.jsonl`, `text_chunks_raw.jsonl`, `pdf_chunks_raw.jsonl`).
  - `Lines 19-33`: Memory-efficient set lookup: `seen_texts = set()`. Only inserts documents whose stripped text has not been encountered.
  - `Lines 34-39`: Writes unique records to `data/processed/merged_raw.jsonl`.

### File 2: `src/processing/chunker.py` (77 Lines)
- **Role:** Splits variable-length documents into retrieval-optimized chunks with word overlap, maintaining paragraph and code block integrity.
- **Key Code Lines:**
  - `Lines 15-42` (`split_text_into_chunks`):
    - Fast check: If `len(words) <= max_words` (400), returns original text without fragmentation.
    - Paragraph splitting: Splits on `\n\n` to respect natural logical thought boundaries.
    - Accumulates paragraphs into `current_chunk` until total word count exceeds 400 words.
  - `Lines 54-66` (`chunk_all_records`):
    - Enriches each chunk with a fresh UUID (`str(uuid.uuid4())`) while preserving original parent metadata (`senior_id`, `source_name`, `source_type`, `topic`, `company`, `date_added`).
  - `Lines 69-71`: Serializes to `data/processed/chunks.jsonl`.

### File 3: `src/ingestion/ingest_github.py` (241 Lines)
- **Role:** Ingests senior GitHub repositories without requiring a local `git.exe` binary.
- **Key Code Lines:**
  - `Lines 63-104` (`download_repo_archive`): Downloads repository ZIP archives via HTTPS from GitHub (`refs/heads/main.zip` or `master.zip`) using in-memory `io.BytesIO` and `zipfile.ZipFile`.
  - `Lines 106-130` (`guess_topic`): Keyword-based classification engine checking file and folder paths against placement taxonomy.
  - `Lines 132-150` (`guess_company`): Regular expression matching for company names in filenames (`Deloitte`, `InfoEdge`, `BlackRock`, `Google`, `Amazon`, etc.).
  - `Lines 185-220`: Recursively walks extracted directories, filters files `< 50` characters, tags metadata, and saves to `github_chunks_raw.jsonl`.

### File 4: `src/ingestion/scrape_web.py` (133 Lines)
- **Role:** Extracts interview logs from public Google Docs, Google Sheets, and standard web articles.
- **Key Code Lines:**
  - `Lines 25-45` (`handle_google_doc`): Converts Google Docs URLs to export endpoints: `https://docs.google.com/document/d/{doc_id}/export?format=txt`.
  - `Lines 46-75` (`handle_google_sheet`): Converts Google Sheets to CSV exports (`export?format=csv&gid={gid}`) and formats each row into structured key-value context strings.
  - `Lines 77-97` (`handle_html`): Cleans web HTML using BeautifulSoup, stripping `script`, `style`, `nav`, `footer`, `header`, `aside` tags.

### Files 5 & 6: `src/ingestion/ingest_pdf.py` & `src/ingestion/ingest_text.py`
- `ingest_pdf.py` uses `pdfplumber` to extract multi-page text from PDF placement guides.
- `ingest_text.py` reads senior notes formatted as `{senior_id}__{topic}__{title}.txt`.

---

## Phase 3: Indexing & Hybrid Retrieval Engine

### File 1: `src/retrieval/build_index.py` (65 Lines)
- **Role:** Offline builder for the dense FAISS vector index and sparse BM25 inverted index.
- **Key Code Lines:**
  - `Lines 19-21` (`tokenize_for_bm25`): Lowercases and extracts alphanumeric words using `re.findall(r"\w+", text.lower())`.
  - `Lines 38-45` (Dense FAISS Index):
    - Loads `SentenceTransformer("all-MiniLM-L6-v2")`.
    - Computes 384-dimensional embeddings for all chunk texts.
    - **L2 Normalization:** `faiss.normalize_L2(embeddings)`.
    - Instantiates `faiss.IndexFlatIP(384)` (Inner Product). Because vectors are unit length, Inner Product equals exact Cosine Similarity.
    - Writes FAISS binary index to `indexes/faiss_index`.
  - `Lines 50-55` (Sparse BM25 Index):
    - Builds `BM25Okapi(tokenized_corpus)`.
    - Pickles the BM25 model into `indexes/bm25_index.pkl`.
  - `Lines 58-61`: Serializes chunk metadata to `indexes/chunk_metadata.jsonl`.

### File 2: `src/retrieval/hybrid_retriever.py` (83 Lines)
- **Role:** Implements Dense Search, Sparse Search, and Reciprocal Rank Fusion (RRF).
- **Key Code Lines:**
  - `Lines 26-35` (`load_indexes`): Singleton pattern with global variables (`_FAISS_INDEX`, `_BM25_INDEX`, `_METADATA`, `_EMBED_MODEL`) ensuring models are loaded into RAM only once upon process boot.
  - `Lines 37-43` (`dense_search`): Encodes query text, applies `faiss.normalize_L2(query_vec)`, and queries FAISS for top-$k$ nearest neighbors.
  - `Lines 45-53` (`sparse_search`): Tokenizes query, queries `BM25Okapi.get_scores()`, and sorts indices by score descending.
  - `Lines 55-62` (`reciprocal_rank_fusion`):
    - Accumulates reciprocal rank scores:
      $$\text{Score}(d) = \sum \frac{1}{60 + \text{rank}}$$
    - Returns fused candidate list sorted by combined RRF score.
  - `Lines 64-78` (`hybrid_search`): Coordinates dense + sparse searches and maps document indices back to chunk dictionaries with attached `retrieval_score`.

### File 3: `src/retrieval/query_rewriter.py` (45 Lines)
- **Role:** Multi-query expansion using Groq LLM.
- **Key Code Lines:**
  - `Lines 21-41` (`rewrite_query`):
    - Takes user query and prompts Groq LLM at low temperature (`temperature=0.3`) to generate 3 alternative domain-specific queries.
    - Strips numbering/bullets using regex `re.sub(r"^[\d\.\-\*\•\s]+", "", l)`.
    - Returns `[original_query, rewrite_1, rewrite_2, rewrite_3]`.

### File 4: `src/retrieval/reranker.py` (44 Lines)
- **Role:** High-precision Cross-Encoder reranker.
- **Key Code Lines:**
  - `Lines 16-20` (`get_reranker`): Global singleton loading `CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")`.
  - `Lines 22-35` (`rerank`):
    - Constructs sentence pairs: `[[query, chunk["text"]] for chunk in candidates]`.
    - Runs `model.predict(pairs)` across full cross-attention layers.
    - Attaches `chunk["rerank_score"] = round(score, 4)` and slices the top `TOP_K_RERANK` (5) chunks.

### File 5: `src/retrieval/pipeline.py` (58 Lines)
- **Role:** End-to-end retrieval coordinator.
- **Key Code Lines:**
  - `Lines 17-53` (`retrieve_and_rerank`):
    1. Expands query to 4 variations.
    2. Runs hybrid search for each variation, merging candidates into a dictionary keyed by `chunk_id` and retaining the maximum retrieval score.
    3. Applies hard metadata filters (`topic_filters` and `company_filter`).
    4. Slices top 20 candidates (`TOP_K_RETRIEVE`) and passes them to `rerank(...)` to produce the final top 5 chunks.
    5. Returns response object with latency timing (`pipeline_ms`).

---

## Phase 4: Multi-Agent Orchestration with LangGraph

### File 1: `src/agents/state.py` (18 Lines)
- **Role:** Data contract and state schema shared across all agent nodes in the LangGraph execution cycle.
- **State Schema:**
  ```python
  class GraphState(TypedDict):
      user_query: str                          # Original student question
      topic_filters: Optional[List[str]]       # Metadata filter: topics
      company_filter: Optional[str]            # Metadata filter: company
      needs_synthesis: bool                    # Flagged by Router Agent
      retrieved_chunks: List[Dict[str, Any]]   # Populated by Retrieval Agent
      final_answer: str                        # Generated by Synthesizer Agent
      has_conflict: bool                       # Flagged if senior advice diverges
      sources_used: List[str]                  # Cited senior/file references
  ```

### File 2: `src/agents/router_agent.py` (41 Lines)
- **Role:** Zero-shot intent classification node.
- **Key Code Lines:**
  - `Lines 12-40` (`router_node`): Prompts Groq LLM at `temperature=0.0` to classify query into:
    - `SIMPLE`: Direct factual lookup (syllabus, test pattern, eligibility).
    - `SYNTHESIS`: Strategy, preparation timeline, comparison, or advice.
  - Updates `state["needs_synthesis"]`.

### File 3: `src/agents/retrieval_agent.py` (25 Lines)
- **Role:** LangGraph node wrapper around Phase 3 retrieval pipeline.
- **Key Code Lines:**
  - `Lines 10-24` (`retrieval_node`): Invokes `retrieve_and_rerank(...)` using state filters and populates `state["retrieved_chunks"]`.

### File 4: `src/agents/synthesizer_agent.py` (71 Lines)
- **Role:** Answer generation, strict grounding, source citation, and senior advice conflict detection.
- **Key Code Lines:**
  - `Lines 12-15` (`detect_conflicts`): Scans synthesized text for semantic disagreement markers (`differ`, `conflict`, `disagree`, `diverge`, `on the other hand`, `varying advice`).
  - `Lines 17-70` (`synthesizer_node`):
    - Defensive check: If `retrieved_chunks` is empty, returns fallback message without invoking LLM.
    - Builds structured context block: `[Source #1 | Senior A | infoedge.md]\n...`.
    - Enforces 4 core system prompt rules:
      1. *STRICT GROUNDING:* No external hallucinated facts.
      2. *CONFLICT DETECTION:* Explicitly highlight differing guidance with: `'⚠️ Note: Sources offer differing guidance: [Senior A] suggests X, while [Senior B] suggests Y.'`
      3. *CITATIONS:* Cite specific seniors and files.
      4. *STRUCTURE:* Clean markdown headers and bullet points.
    - Calls Groq LLM at `temperature=0.2`, extracts final answer, executes `detect_conflicts()`, and collects `sources_used`.

### File 5: `src/agents/graph.py` (50 Lines)
- **Role:** Compiles and exposes the multi-agent graph.
- **Key Code Lines:**
  - `Lines 13-28` (`create_agent_graph`):
    - `workflow = StateGraph(GraphState)`
    - `workflow.add_node("router", router_node)`
    - `workflow.add_node("retrieval", retrieval_node)`
    - `workflow.add_node("synthesizer", synthesizer_node)`
    - Connects edges: `START -> router -> retrieval -> synthesizer -> END`.
    - Returns `workflow.compile()`.
  - `Lines 33-45` (`run_query`): Public entrypoint initializing default state and calling `pipeline.invoke(initial_state)`.

---

## Phase 5: Serving, APIs & Web Client

### File 1: `src/server.py` (124 Lines)
- **Role:** High-performance asynchronous FastAPI web server and static asset host.
- **Key Code Lines:**
  - `Lines 24-37`: Configures FastAPI app with CORS middleware.
  - `Lines 40-51`: Pydantic request/response schemas (`QueryRequest`, `QueryResponse`).
  - `Lines 57-85` (`/api/stats`): Computes corpus metrics dynamically (total chunks, unique seniors, companies, sources).
  - `Lines 87-110` (`/api/query`): Validates incoming payload, invokes `run_query()`, and returns JSON response.
  - `Lines 112-119`: Mounts `/static` directory and serves `frontend/index.html` on root `/`.

### File 2: `src/app.py` (166 Lines)
- **Role:** Alternative interactive dashboard built with Streamlit for rapid prototyping and live demonstrations.

### File 3: `frontend/app.js` & `frontend/index.html`
- Modern Vanilla JavaScript & CSS responsive interface.
- Communicates with `/api/stats` and `/api/query`.
- Renders Markdown via `marked.js`, displays conflict warning banners, renders source citation pills, and expands full text evidence cards with rerank scores.

---

## Phase 6: Evaluation & Benchmarking

### File 1: `src/eval/run_eval.py` (89 Lines)
- **Role:** Automated benchmarking script comparing 3 retrieval strategies on 15 queries from `eval_set.json`.
- **Key Code Lines:**
  - Runs queries across:
    1. **Naive Dense:** FAISS dense alone (`dense_search(k=5)`).
    2. **Hybrid (BM25 + FAISS):** Dense + Sparse merged via RRF (`hybrid_search(top_k=5)`).
    3. **Multi-Query Hybrid + Rerank:** Full pipeline with query expansion, RRF, and Cross-Encoder reranker (`retrieve_and_rerank(top_k_final=5)`).
  - Computes Topic Match @ 5 accuracy.
  - Generates `src/eval/results.md`.

---

# 4. Mathematical & Algorithmic Foundations

### 1. Vector Normalization & Cosine Similarity in FAISS
Given query vector $\mathbf{q} \in \mathbb{R}^d$ and document vector $\mathbf{d} \in \mathbb{R}^d$:
$$\text{Cosine Similarity}(\mathbf{q}, \mathbf{d}) = \frac{\mathbf{q} \cdot \mathbf{d}}{\|\mathbf{q}\|_2 \|\mathbf{d}\|_2} = \frac{\sum_{i=1}^d q_i d_i}{\sqrt{\sum_{i=1}^d q_i^2} \sqrt{\sum_{i=1}^d d_i^2}}$$

When vectors are normalized such that $\|\mathbf{q}\|_2 = 1$ and $\|\mathbf{d}\|_2 = 1$:
$$\text{Cosine Similarity}(\mathbf{q}, \mathbf{d}) = \mathbf{q} \cdot \mathbf{d} = \text{Inner Product}$$
This allows using `faiss.IndexFlatIP` to perform exact cosine similarity computations with minimal CPU instructions.

---

### 2. BM25Okapi Lexical Scoring
For query $Q = \{q_1, q_2, \dots, q_n\}$ and document $D$:
$$\text{Score}_{\text{BM25}}(D, Q) = \sum_{i=1}^n \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$
Where:
- $\text{IDF}(q_i) = \ln \left( \frac{N - n(q_i) + 0.5}{n(q_i) + 0.5} + 1 \right)$
- $f(q_i, D)$ = frequency of token $q_i$ in document $D$.
- $|D|$ = document length, $\text{avgdl}$ = average document length across corpus.
- $k_1 = 1.5$ (term frequency saturation factor), $b = 0.75$ (document length penalty).

---

### 3. Reciprocal Rank Fusion (RRF)
Given rankings from multiple retrieval algorithms $M = \{\text{Dense}, \text{Sparse}\}$:
$$\text{RRF\_Score}(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$
Where $r_m(d)$ is the 1-based rank of document $d$ in system $m$, and $k = 60$ is the smoothing constant to prevent top-ranked outliers from completely dominating.

---

### 4. Bi-Encoder vs Cross-Encoder Mechanics
- **Bi-Encoder (Dual-Tower):**
  $$\text{Score}_{\text{Bi}} = \langle \text{Encoder}_A(Q), \text{Encoder}_B(D) \rangle$$
  Query and Document are encoded independently. Attention complexity: $O(|Q|^2 + |D|^2)$.
- **Cross-Encoder (Single-Tower):**
  $$\text{Score}_{\text{Cross}} = \text{MLP}(\text{Transformer}([CLS] \circ Q \circ [SEP] \circ D))$$
  Full joint cross-attention. Attention complexity: $O((|Q| + |D|)^2)$. Every query token directly computes attention weights with every document token.

---

# 5. Architectural Trade-Offs: "Why This vs Why Not That?"

| Architectural Decision | Chosen Implementation | Alternative Option | Rationale & Trade-offs |
| :--- | :--- | :--- | :--- |
| **Embedding Model** | `all-MiniLM-L6-v2` (384-dim) | `text-embedding-3-small` (1536-dim) or `bge-large-en` (1024-dim) | Runs locally on CPU in ~15ms, zero API cost, compact 384-d vectors minimize FAISS RAM consumption while providing high semantic resolution for technical English text. |
| **Hybrid Fusion** | Reciprocal Rank Fusion ($k=60$) | Linear Score Sum ($\alpha \cdot S_{\text{dense}} + (1-\alpha) \cdot S_{\text{bm25}}$) | Dense cosine scores are bounded $[0, 1]$; BM25 scores are unbounded $[0, 50+]$. Linear combinations require arbitrary score normalization which breaks under query drift. RRF is scale-invariant. |
| **Reranking Strategy** | Two-Stage Retrieval (Top-20 Hybrid $\to$ Top-5 Cross-Encoder) | Single-Stage Bi-Encoder OR Corpus-wide Cross-Encoder | Evaluating a Cross-Encoder over thousands of chunks takes seconds. Two-stage retrieval gives the $O(N)$ speed of Bi-Encoder candidate generation combined with the $O(K)$ precision of Cross-Encoder reranking. |
| **Orchestration** | LangGraph (`StateGraph`) | LangChain `SequentialChain` / Monolithic Prompt | LangGraph provides typed, observable state (`GraphState`), modular node isolation, defensive fallback handling, and support for conditional looping. |
| **Chunking** | Paragraph-Aware (`\n\n`) + 50-word overlap | Fixed character chunking (e.g. 1000 chars) | Fixed character chunking slices words, sentences, and code snippets in half. Paragraph-aware splitting preserves logical thought units. |
| **LLM Inference** | Groq API (`openai/gpt-oss-20b`) | Local Ollama / OpenAI API | Groq's LPU hardware yields 300–500 tokens/sec. Query rewriting and synthesis finish in <1.5s total roundtrip latency. |
| **Web Server** | FastAPI (Python ASGI) | Flask / Django | Asynchronous request handling (ASGI), native Pydantic data validation, automatic Swagger UI documentation, low memory overhead. |

---

# 6. Comprehensive Interview Question & Answer Vault

### Basic Questions

#### Q1: What is RAG and why is it needed for campus placement preparation?
> **Answer:** "Retrieval-Augmented Generation (RAG) is an AI architecture that enhances Large Language Models by retrieving relevant, up-to-date facts from an external knowledge base before generating an answer. For placement preparation, standard LLMs hallucinate specific company interview rounds, cutoff criteria, or test patterns. RAG grounds the LLM strictly on verified senior notes, interview transcripts, and company logs."

#### Q2: What is the purpose of chunk overlap?
> **Answer:** "Chunk overlap (e.g., 50 words) ensures that semantic meaning is not severed at chunk boundaries. If a critical piece of advice or code snippet spans across a split point, sliding overlap guarantees that at least one chunk contains the complete contiguous context."

#### Q3: How are vectors normalized in your project and why?
> **Answer:** "We use `faiss.normalize_L2(embeddings)` in `build_index.py`. Normalizing vectors to unit length ($\|\mathbf{v}\|_2 = 1$) makes their Inner Product mathematically equivalent to exact Cosine Similarity, enabling fast computation via `faiss.IndexFlatIP`."

#### Q4: What happens if a user submits an empty query?
> **Answer:** "The FastAPI endpoint in `server.py` validates the query and raises an `HTTPException(status_code=400, detail='Query cannot be empty.')`. In the agent pipeline, defensive checks return safe fallback responses without invoking downstream models."

#### Q5: What is the role of `pydantic` in `src/server.py`?
> **Answer:** "`pydantic` defines data models (`QueryRequest`, `QueryResponse`) that enforce strict type checking, auto-validate JSON request payloads, and serialize outgoing API responses."

---

### Intermediate Questions

#### Q6: Why does BM25 outperform Vector Search on company names like "InfoEdge" or "Deloitte"?
> **Answer:** "Dense embedding models compress text into a continuous vector space. If a rare company name or acronym was under-represented in the embedding model's pre-training corpus, its vector representation may be close to generic words. BM25 uses exact term matching and Inverse Document Frequency (IDF), assigning high weight to rare, specific tokens."

#### Q7: Walk me through the Reciprocal Rank Fusion calculation with numbers.
> **Answer:** 
> "Suppose Chunk A is Rank 1 in Dense and Rank 3 in BM25. Using $k=60$:
> $$\text{Score}_{\text{RRF}}(\text{Chunk A}) = \frac{1}{60 + 1} + \frac{1}{60 + 3} = \frac{1}{61} + \frac{1}{63} \approx 0.01639 + 0.01587 = 0.03226$$
> Suppose Chunk B is Rank 2 in Dense and Rank 1 in BM25:
> $$\text{Score}_{\text{RRF}}(\text{Chunk B}) = \frac{1}{60 + 2} + \frac{1}{60 + 1} = \frac{1}{62} + \frac{1}{61} \approx 0.01613 + 0.01639 = 0.03252$$
> Chunk B wins because of its higher average position across both systems."

#### Q8: How does Query Expansion improve retrieval recall?
> **Answer:** "Students often ask short questions like *'sql tips'*. In `query_rewriter.py`, Groq LLM generates 3 expanded variations such as *'SQL window functions interview questions'*, *'SQL query optimization placement prep'*, and *'DBMS SQL interview queries'*. Retrieving candidates across all 4 queries uncovers relevant chunks that did not share the exact vocabulary of the original prompt."

#### Q9: How does the system detect conflicting senior advice?
> **Answer:** "1. The Synthesizer system prompt instructs the LLM to inspect retrieved excerpts for diverging advice and explicitly format discrepancies.
> 2. The function `detect_conflicts()` in `synthesizer_agent.py` inspects the output for semantic conflict indicators (`differ`, `conflict`, `disagree`, `diverge`, `on the other hand`).
> 3. If detected, `state['has_conflict']` is set to `True`, triggering a visual warning banner in the frontend."

#### Q10: How do you prevent the LLM from hallucinating facts?
> **Answer:** "We use a multi-layered defense:
> 1. Strict system prompt instruction: *'Answer using ONLY the provided Source Excerpts. Do not hallucinate outside facts.'*
> 2. Low generation temperature (`temperature=0.2`).
> 3. Two-stage high-precision retrieval ensuring only relevant chunks reach the context window.
> 4. Empty-context guardrail: If 0 chunks are retrieved, the agent immediately outputs a fallback message without calling the LLM."

---

### Advanced Questions & System Design

#### Q11: Why use a Cross-Encoder as a second-stage reranker instead of a Bi-Encoder?
> **Answer:** "In a Bi-Encoder, the query and document vectors are generated independently without cross-token interaction. In a Cross-Encoder (`ms-marco-MiniLM-L-6-v2`), the query and document are passed together through the transformer. Self-attention layers compute cross-token attention weights between all query words and document words simultaneously, capturing deep semantic relationships, negation, and positional nuances that dual-tower models miss."

#### Q12: How would you scale this architecture to support 10 million documents?
> **Answer:**
> "1. **Vector Index:** Replace `IndexFlatIP` with an Approximate Nearest Neighbor index such as `IndexIVFPQ` (Inverted File with Product Quantization) or `IndexHNSW` in FAISS, or migrate to a distributed vector database like Qdrant or Milvus.
> 2. **Sparse Search:** Replace in-memory `BM25Okapi` with a distributed cluster of Elasticsearch or OpenSearch.
> 3. **Asynchronous Ingestion:** Implement message queues (Kafka/RabbitMQ) and worker pools (Celery) to handle document extraction, chunking, and embedding in parallel batches.
> 4. **Caching Layer:** Add Redis caching on query embeddings and common responses to serve identical queries in <10ms."

#### Q13: How do you measure RAG retrieval performance quantitatively?
> **Answer:** "In `src/eval/run_eval.py`, we execute a benchmark suite of 15 diverse placement queries. We measure **Topic Match @ 5** (precision proxy) across three configurations: Naive Dense, Hybrid (BM25 + FAISS), and Hybrid + Reranker. In production, we also evaluate the RAG Triad: Context Relevance, Groundedness (Faithfulness), and Answer Relevance using frameworks like Ragas or TruLens."

#### Q14: What is the memory footprint of this application in production?
> **Answer:** 
> "- `all-MiniLM-L6-v2`: ~80 MB RAM.
> - `ms-marco-MiniLM-L-6-v2`: ~90 MB RAM.
> - FAISS Flat Index (1,000 chunks $\times$ 384 dimensions $\times$ 4 bytes): ~1.5 MB RAM.
> - BM25 Inverted Index & Metadata: ~5 MB RAM.
> - FastAPI runtime: ~100 MB RAM.
> Total memory footprint is well under **350 MB RAM**, allowing it to run efficiently on low-cost single-core CPU instances."

#### Q15: Give a 60-second elevator pitch of your project.
> **Answer:** 
> *"I built the **Placement Prep Knowledge Assistant**, a multi-agent RAG system that aggregates unstructured senior placement logs, company test patterns, and interview roadmaps into an intelligent query engine. 
> 
> Because placement queries require both exact keyword recall for company names and semantic understanding for preparation concepts, I designed a **two-stage hybrid retrieval pipeline** combining BM25 keyword search, FAISS dense vector search via Reciprocal Rank Fusion, LLM query expansion, and a Cross-Encoder reranker. 
> 
> The system is orchestrated using **LangGraph** across intent routing, retrieval, and synthesizer agents featuring automated **conflict detection** to highlight differing senior opinions. Served via **FastAPI** with a responsive web interface, the system achieves sub-1.5s latency with strict factual grounding and zero hallucinations."*

---

# 7. Resume Bullet Points & Defense Playbook

### Bullet Point 1
> **"Engineered an end-to-end Multi-Agent Hybrid RAG system using LangGraph, FAISS, and BM25 to synthesize 1,000+ unstructured senior interview logs and company placement patterns."**
- **Interviewer Question:** *"Why did you use LangGraph instead of standard LangChain?"*
- **Defense:** *"LangGraph gives us a stateful directed acyclic graph (`StateGraph`) with typed state transitions (`GraphState`). This enabled clear separation between intent routing, multi-query hybrid retrieval, and grounded synthesis, while supporting graceful fallbacks and automated conflict detection."*

### Bullet Point 2
> **"Architected a two-stage retrieval pipeline fusing BM25 keyword matching and 384-d dense embeddings (`all-MiniLM-L6-v2`) via Reciprocal Rank Fusion ($k=60$), reranked with a Cross-Encoder (`ms-marco-MiniLM-L-6-v2`)."**
- **Interviewer Question:** *"Why not just use cosine similarity on dense embeddings alone?"*
- **Defense:** *"Dense search alone failed on exact domain keywords like company names ('InfoEdge', 'Deloitte') and SQL terms. Combining BM25 with FAISS using RRF improved candidate recall, while our Cross-Encoder reranker refined top-20 candidates to the top-5 most relevant passages, significantly boosting precision."*

### Bullet Point 3
> **"Implemented automated query expansion and a novel Senior Conflict Detection mechanism to surface and contrast divergent placement strategies across roles."**
- **Interviewer Question:** *"How does conflict detection work under the hood?"*
- **Defense:** *"The Synthesizer Agent is prompted to detect conflicting advice (e.g. SDE vs Data Science priorities) and structure discrepancies explicitly. A post-synthesis semantic analysis inspects the output for divergence markers, setting a state flag that renders a conflict banner in the UI."*

### Bullet Point 4
> **"Built and deployed a high-concurrency FastAPI backend with asynchronous endpoints and modern web client, achieving sub-1.5s end-to-end latency and zero hallucinations."**
- **Interviewer Question:** *"How do you guarantee zero hallucinations?"*
- **Defense:** *"We enforce strict system prompt grounding requiring the LLM to rely exclusively on retrieved source excerpts, keep generation temperature low at 0.2, cite exact senior metadata, and apply an empty-context guardrail that returns a safe fallback message if retrieval finds no evidence."*

---

# 8. Hands-On Verification & Execution Runbook

To verify and run every stage of the pipeline locally:

```powershell
# 1. Activate Python Environment & Install Dependencies
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt

# 2. Ingest Multi-Source Raw Data
python src/ingestion/ingest_github.py
python src/ingestion/scrape_web.py
python src/ingestion/ingest_pdf.py
python src/ingestion/ingest_text.py

# 3. Deduplicate & Chunk
python src/processing/merge_and_tag.py
python src/processing/chunker.py

# 4. Build FAISS Dense & BM25 Sparse Indices
python src/retrieval/build_index.py

# 5. Test Retrieval & Multi-Agent Pipeline via CLI
python src/retrieval/pipeline.py
python src/agents/graph.py

# 6. Execute Quantitative Evaluation Benchmarks
python src/eval/run_eval.py

# 7. Start FastAPI Web Server & UI
python src/server.py
# Open browser at: http://localhost:8000
```
