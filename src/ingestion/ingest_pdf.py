"""
PDF Ingestion Module
Extracts text from PDF interview guides using pdfplumber.
"""

import os
import uuid
import datetime
import json
from pathlib import Path
import pdfplumber
import src.config as config
from src.ingestion.ingest_github import guess_topic, guess_company

def extract_pdf_text(filepath: Path) -> str:
    """Extracts text from all pages of a PDF."""
    page_texts = []
    try:
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text and text.strip():
                    page_texts.append(text.strip())
    except Exception:
        return ""
    return "\n\n".join(page_texts)

def ingest_pdf_files():
    records = []
    
    # Search in raw data directory for any PDFs
    for pdf_file in config.RAW_DIR.rglob("*.pdf"):
        if ".git" in pdf_file.parts:
            continue
        text = extract_pdf_text(pdf_file)
        if len(text.strip()) >= 50:
            records.append({
                "chunk_id": str(uuid.uuid4()),
                "text": text.strip(),
                "source_type": "pdf",
                "source_name": pdf_file.name,
                "senior_id": "senior_pdf_contributor",
                "topic": guess_topic(pdf_file.name, "general"),
                "company": guess_company(pdf_file.name),
                "date_added": datetime.date.today().isoformat()
            })

    out_file = config.PROCESSED_DIR / "pdf_chunks_raw.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"Ingested {len(records)} PDF documents -> {out_file}")

if __name__ == "__main__":
    ingest_pdf_files()
