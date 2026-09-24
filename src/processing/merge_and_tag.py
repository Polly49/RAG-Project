"""
Raw Source Merger & Deduplicator
Combines raw JSONL files from GitHub, text notes, web, and PDFs into one deduplicated dataset.
"""

import json
from pathlib import Path
import src.config as config

def merge_raw_sources():
    raw_files = [
        config.PROCESSED_DIR / "github_chunks_raw.jsonl",
        config.PROCESSED_DIR / "text_chunks_raw.jsonl",
        config.PROCESSED_DIR / "web_chunks_raw.jsonl",
        config.PROCESSED_DIR / "pdf_chunks_raw.jsonl"
    ]

    all_records = []
    seen_texts = set()

    for file_path in raw_files:
        if not file_path.exists():
            continue
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                text = rec.get("text", "").strip()
                if text and text not in seen_texts:
                    seen_texts.add(text)
                    all_records.append(rec)

    merged_file = config.PROCESSED_DIR / "merged_raw.jsonl"
    with open(merged_file, "w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec) + "\n")

    print(f"Merged {len(all_records)} unique documents -> {merged_file}")

if __name__ == "__main__":
    merge_raw_sources()
