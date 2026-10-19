"""
Text Notes Ingestion
Reads .txt notes from data/raw/text_notes/ and converts them to JSONL format.
"""

import os
import uuid
import datetime
import json
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.config as config

def parse_filename(filename: str) -> tuple[str, str]:
    """Extracts senior_id and topic from filenames like 'sameer__sql__window_functions.txt'."""
    parts = Path(filename).stem.split("__")
    if len(parts) >= 2:
        senior_id = parts[0].strip()
        topic = parts[1].strip().lower() if parts[1].strip().lower() in config.TOPICS else "general"
        return senior_id, topic
    return "unknown", "general"

def ingest_text_notes():
    notes_dir = config.RAW_DIR / "text_notes"
    if not notes_dir.exists():
        print(f"Directory not found: {notes_dir}")
        return

    records = []
    for file_path in notes_dir.glob("*.txt"):
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read().strip()
        if len(content) < 30:
            continue

        senior_id, topic = parse_filename(file_path.name)
        records.append({
            "chunk_id": str(uuid.uuid4()),
            "text": content,
            "source_type": "text",
            "source_name": file_path.name,
            "senior_id": senior_id,
            "topic": topic,
            "company": None,
            "date_added": datetime.date.today().isoformat()
        })

    out_file = config.PROCESSED_DIR / "text_chunks_raw.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"Ingested {len(records)} text notes -> {out_file}")

if __name__ == "__main__":
    ingest_text_notes()
