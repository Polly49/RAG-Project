"""
Web & Google Workspace Ingestion
Scrapes text from Google Docs, Google Sheets, and placement preparation websites.
"""

import os
import uuid
import datetime
import json
import re
import csv
import io
import urllib.request
from bs4 import BeautifulSoup
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.config as config

def fetch_url(url: str, timeout: int = 15) -> str:
    """Fetches raw webpage or export text."""
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")

def handle_google_doc(url: str, topic: str, company: str | None) -> list[dict]:
    """Exports Google Doc as plain text."""
    match = re.search(r"/document/d/([a-zA-Z0-9-_]+)", url)
    if not match:
        return []
    doc_id = match.group(1)
    export_url = f"https://docs.google.com/document/d/{doc_id}/export?format=txt"
    text = fetch_url(export_url).strip()
    if len(text) < 50:
        return []
    return [{
        "chunk_id": str(uuid.uuid4()),
        "text": text,
        "source_type": "web",
        "source_name": f"GoogleDoc_{doc_id}",
        "senior_id": "senior_doc_contributor",
        "topic": topic,
        "company": company,
        "date_added": datetime.date.today().isoformat()
    }]

def handle_google_sheet(url: str, topic: str, company: str | None) -> list[dict]:
    """Exports Google Sheet as CSV and converts rows to interview records."""
    match = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", url)
    if not match:
        return []
    sheet_id = match.group(1)
    gid_match = re.search(r"[#?&]gid=([0-9]+)", url)
    gid = gid_match.group(1) if gid_match else "0"
    
    csv_text = fetch_url(f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}")
    reader = list(csv.reader(io.StringIO(csv_text)))
    if len(reader) < 2:
        return []

    headers = [h.strip() for h in reader[0]]
    records = []
    for row in reader[1:]:
        row_text = "\n".join([f"{headers[i]}: {cell.strip()}" for i, cell in enumerate(row) if i < len(headers) and cell.strip()])
        if len(row_text) >= 30:
            records.append({
                "chunk_id": str(uuid.uuid4()),
                "text": row_text,
                "source_type": "web",
                "source_name": f"GoogleSheet_{sheet_id}_gid_{gid}",
                "senior_id": "sheet_contributor",
                "topic": topic if topic != "general" else "hr_interview",
                "company": company,
                "date_added": datetime.date.today().isoformat()
            })
    return records

def handle_html(url: str, topic: str, company: str | None) -> list[dict]:
    """Extracts article text from regular HTML websites."""
    html = fetch_url(url)
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    
    body = soup.find("article") or soup.find("main") or soup.body
    text = body.get_text(separator="\n", strip=True) if body else ""
    if len(text) < 100:
        return []
    return [{
        "chunk_id": str(uuid.uuid4()),
        "text": text,
        "source_type": "web",
        "source_name": url.split("//")[-1].rstrip("/"),
        "senior_id": None,
        "topic": topic,
        "company": company,
        "date_added": datetime.date.today().isoformat()
    }]

def scrape_web_sources():
    sources_file = config.RAW_DIR / "web_sources.txt"
    if not sources_file.exists():
        return

    all_records = []
    with open(sources_file, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]

    for line in lines:
        parts = [p.strip() for p in line.split(",")]
        url = parts[0]
        company = parts[1] if len(parts) > 1 and parts[1] else None
        topic = parts[2] if len(parts) > 2 and parts[2] in config.TOPICS else "general"

        try:
            if "docs.google.com/document" in url:
                all_records.extend(handle_google_doc(url, topic, company))
            elif "docs.google.com/spreadsheets" in url:
                all_records.extend(handle_google_sheet(url, topic, company))
            else:
                all_records.extend(handle_html(url, topic, company))
        except Exception as e:
            print(f"Error fetching {url}: {e}")

    out_file = config.PROCESSED_DIR / "web_chunks_raw.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r) + "\n")

    print(f"Ingested {len(all_records)} web records -> {out_file}")

if __name__ == "__main__":
    scrape_web_sources()
