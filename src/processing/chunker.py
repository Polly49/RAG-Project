"""
Text Chunker Module
Splits merged placement documents into overlapping semantic chunks using RecursiveCharacterTextSplitter.
"""

import sys
import json
import uuid
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.config as config

# Industry-standard production text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,      # ~300 words, fits comfortably within embedding model context
    chunk_overlap=200,     # ~50 words overlap to preserve context across boundaries
    separators=["\n\n", "\n", " ", ""]
)

def chunk_all_records():
    merged_path = config.PROCESSED_DIR / "merged_raw.jsonl"
    if not merged_path.exists():
        print(f"File not found: {merged_path}. Run merge_and_tag.py first.")
        return

    with open(merged_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    all_chunks = []
    for rec in records:
        text = rec.get("text", "").strip()
        if not text:
            continue
            
        chunks = text_splitter.split_text(text)
        for chunk_text in chunks:
            all_chunks.append({
                "chunk_id": str(uuid.uuid4()),
                "text": chunk_text,
                "source_type": rec.get("source_type", "unknown"),
                "source_name": rec.get("source_name", "unknown"),
                "senior_id": rec.get("senior_id"),
                "topic": rec.get("topic", "general"),
                "company": rec.get("company"),
                "date_added": rec.get("date_added")
            })

    output_path = Path(config.CHUNKS_FILE)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c) + "\n")

    print(f"Created {len(all_chunks)} chunks from {len(records)} documents -> {output_path}")

if __name__ == "__main__":
    chunk_all_records()
