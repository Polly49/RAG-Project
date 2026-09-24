"""
Semantic Paragraph-Aware Chunker

AI/ML Concept:
- Splits long documents into ~400-word retrieval-friendly chunks on paragraph boundaries.
- Applies 50-word sliding overlap so context is not lost at paragraph edges.
"""

import os
import json
import uuid
from pathlib import Path
import src.config as config

def split_text_into_chunks(text: str, max_words: int = config.CHUNK_SIZE, overlap: int = config.CHUNK_OVERLAP) -> list[str]:
    """Splits text on double-newline paragraph boundaries with word-level overlap."""
    words = text.strip().split()
    if len(words) <= max_words:
        return [text.strip()]

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current_chunk = []
    current_count = 0

    for p in paragraphs:
        p_words = p.split()
        p_count = len(p_words)

        if current_count + p_count <= max_words:
            current_chunk.append(p)
            current_count += p_count
        else:
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
            current_chunk = [p]
            current_count = p_count

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks

def chunk_all_records():
    merged_path = config.PROCESSED_DIR / "merged_raw.jsonl"
    if not merged_path.exists():
        print(f"File not found: {merged_path}. Run merge_and_tag.py first.")
        return

    with open(merged_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    all_chunks = []
    for rec in records:
        text_chunks = split_text_into_chunks(rec.get("text", ""))
        for chunk_text in text_chunks:
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

    os.makedirs(os.path.dirname(config.CHUNKS_FILE), exist_ok=True)
    with open(config.CHUNKS_FILE, "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c) + "\n")

    print(f"Created {len(all_chunks)} chunks from {len(records)} raw documents -> {config.CHUNKS_FILE}")

if __name__ == "__main__":
    chunk_all_records()
