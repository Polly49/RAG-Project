"""
Configuration & Hyperparameters for Placement Prep RAG
Centralizes all AI/ML models, paths, and retrieval settings.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# --- Project Path Setup ---
# Automatically add project root to Python path for clean imports across all modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load .env file
load_dotenv(PROJECT_ROOT / ".env")

# Reconfigure UTF-8 for Windows console (emojis in terminal)
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# --- Directory Paths ---
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
INDEXES_DIR = PROJECT_ROOT / "indexes"

# Files
CHUNKS_FILE = str(PROCESSED_DIR / "chunks.jsonl")
FAISS_INDEX_PATH = str(INDEXES_DIR / "faiss_index")
BM25_INDEX_PATH = str(INDEXES_DIR / "bm25_index.pkl")
METADATA_PATH = str(INDEXES_DIR / "chunk_metadata.jsonl")

# GitHub Ingestion Paths
GITHUB_LIST_FILE = RAW_DIR / "github_repos.txt"   # Text file listing repo URLs to ingest
CLONED_REPOS_DIR = RAW_DIR / "cloned_repos"       # Where extracted repo files are stored

# --- AI & ML Hyperparameters ---
# 1. LLM Model (Groq)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DEFAULT_GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# 2. Embedding Model (Bi-Encoder for Dense Semantic Search)
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # 384-dimensional dense vectors

# 3. Cross-Encoder Model (For High-Precision Reranking)
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# 4. Chunking Settings
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 400))       # ~400 words per chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))   # 50 words sliding overlap

# 5. Retrieval Top-K Settings
TOP_K_RETRIEVE = int(os.getenv("TOP_K_RETRIEVE", 20))  # Retrieve top 20 candidates from hybrid search
TOP_K_RERANK = int(os.getenv("TOP_K_RERANK", 5))       # Cross-Encoder picks top 5 final chunks

# Permitted topic categories
TOPICS = [
    "dsa", "sql", "aptitude", "resume",
    "hr_interview", "system_design", "core_cs",
    "genai_ds", "general"
]
