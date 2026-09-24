import os
import sys
import uuid
import datetime
import json
import re
import io
import zipfile  
import urllib.request
import urllib.error

# Add project root to sys.path so we can import src.config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def parse_github_list(filepath):
    """
    Reads the list of senior GitHub repositories from a text file.
    Expected line format: <repo_url>,<senior_id>,<default_topic>
    Lines starting with '#' or empty lines are ignored.
    """
    repos = []
    if not os.path.exists(filepath):
        print(f"[Warning] Repos list file not found at: {filepath}")
        return repos

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) < 3:
                print(f"[Warning] Skipping malformed line: {line}")
                continue
            repo_url = parts[0].strip()
            senior_id = parts[1].strip()
            default_topic = parts[2].strip()
            
            # Verify that default_topic is in permitted topics list
            if default_topic not in config.TOPICS:
                print(f"[Warning] Topic '{default_topic}' not in TOPICS. Falling back to 'general'.")
                default_topic = "general"
                
            repos.append({
                "url": repo_url,
                "senior_id": senior_id,
                "default_topic": default_topic
            })
    return repos

def get_repo_owner_and_name(url):
    """
    Extracts the owner and repository name from a GitHub URL.
    Example: https://github.com/sameerwork826/placements -> ('sameerwork826', 'placements')
    """
    clean_url = url.rstrip("/").removesuffix(".git")
    parts = clean_url.split("/")
    if len(parts) >= 2:
        return parts[-2], parts[-1]
    return "unknown", parts[-1]

def download_repo_archive(owner, repo_name, dest_dir):
    """
    Downloads and extracts a GitHub repository as a ZIP archive.
    Tries 'main' branch first, then falls back to 'master'.
    This avoids needing local git.exe installed on Windows.
    """
    os.makedirs(dest_dir, exist_ok=True)
    branches = ["main", "master"]
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    for branch in branches:
        zip_url = f"https://github.com/{owner}/{repo_name}/archive/refs/heads/{branch}.zip"
        try:
            req = urllib.request.Request(zip_url, headers=headers)
            with urllib.request.urlopen(req, timeout=20) as resp:
                if resp.status == 200:
                    zip_data = resp.read()
                    with zipfile.ZipFile(io.BytesIO(zip_data)) as z:
                        # Extract files, stripping the top-level root folder in the zip
                        for member in z.infolist():
                            # Path inside zip is typically '<repo>-<branch>/subfolder/file.md'
                            parts = member.filename.split("/", 1)
                            if len(parts) > 1 and parts[1]:
                                rel_path = parts[1]
                                target_path = os.path.join(dest_dir, rel_path)
                                if member.is_dir():
                                    os.makedirs(target_path, exist_ok=True)
                                else:
                                    os.makedirs(os.path.dirname(target_path), exist_ok=True)
                                    with open(target_path, "wb") as f_out:
                                        f_out.write(z.read(member.filename))
                    print(f"  -> Successfully fetched archive from branch '{branch}'")
                    return True
        except urllib.error.HTTPError as e:
            if e.code == 404:
                continue # Try next branch
            else:
                print(f"  -> HTTP Error {e.code} while fetching branch '{branch}': {e}")
        except Exception as e:
            print(f"  -> Error fetching branch '{branch}': {e}")

    return False

def guess_topic(filepath, default_topic):
    """
    Infers the topic category (dsa, sql, aptitude, resume, etc.) 
    by checking both the filename and parent folder names.
    """
    normalized = filepath.lower().replace("_", " ").replace("-", " ")
    
    # Topic keyword lookup mapping
    topic_keywords = {
        "sql": ["sql", "dbms", "database", "queries"],
        "dsa": ["dsa", "leetcode", "algorithm", "data structure", "tree", "graph", "dp", "recursion"],
        "aptitude": ["aptitude", "puzzle", "quant", "logical", "reasoning"],
        "resume": ["resume", "cv", "portfolio"],
        "hr_interview": ["hr", "behavioral", "interview", "experience", "exp", "behavior"],
        "system_design": ["system design", "lld", "hld", "architecture"],
        "core_cs": ["core cs", "os", "operating system", "cn", "computer network", "oops", "technical"],
        "genai_ds": ["data science", "machine learning", "ml", "genai", "deep learning", "nlp", "statistics", "stats"]
    }
    
    for topic, keywords in topic_keywords.items():
        for kw in keywords:
            if kw in normalized:
                return topic
                
    return default_topic

def guess_company(filename):
    """
    Detects if the file is a company-specific placement interview log 
    (e.g., Deloitte_India.md, InfoEdge.md, BlackRock.md).
    """
    base_name = os.path.splitext(filename)[0]
    clean_name = re.sub(r'(_India|_Exp|_Experience|_Prep|_Guide)$', '', base_name, flags=re.IGNORECASE)
    
    known_companies = [
        "deloitte", "infoedge", "blackrock", "cba", "hilabs", "lek", "lek_consulting",
        "mastercard", "tcg", "tcg_digital", "google", "amazon", "microsoft", 
        "flipkart", "uber", "goldman", "goldman_sachs", "oracle", "barclays"
    ]
    
    for comp in known_companies:
        if comp in clean_name.lower():
            return clean_name.replace("_", " ").title()
            
    return None

def ingest_github_repos():
    """
    Main function to clone or download repos, extract markdown/text files,
    tag them with metadata, and save to raw JSONL.
    """
    repos = parse_github_list(config.GITHUB_LIST_FILE)
    if not repos:
        print("[Info] No GitHub repositories found to ingest.")
        return

    os.makedirs(config.CLONED_REPOS_DIR, exist_ok=True)
    all_records = []
    repos_processed = 0

    for repo_info in repos:
        url = repo_info["url"]
        senior_id = repo_info["senior_id"]
        default_topic = repo_info["default_topic"]
        owner, repo_name = get_repo_owner_and_name(url)
        clone_dest = os.path.join(config.CLONED_REPOS_DIR, repo_name)

        print(f"\n[Processing] Repository: {owner}/{repo_name} ({url})")
        
        # Download repository files
        success = download_repo_archive(owner, repo_name, clone_dest)
        if not success:
            print(f"[Warning] Could not download repository {url}. Skipping.", file=sys.stderr)
            continue

        repos_processed += 1
        files_extracted = 0

        # Walk the repository and extract text and markdown files
        for root_dir, dirs, files in os.walk(clone_dest):
            # Skip hidden git metadata folder
            if ".git" in dirs:
                dirs.remove(".git")
            
            for file in files:
                if file.endswith((".md", ".txt")):
                    filepath = os.path.join(root_dir, file)
                    rel_path = os.path.relpath(filepath, clone_dest)

                    try:
                        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                            content = f.read()
                    except Exception as e:
                        print(f"[Warning] Could not read file {filepath}: {e}. Skipping.")
                        continue

                    # Filter out empty or trivially short files (under 50 chars)
                    if len(content.strip()) < 50:
                        continue

                    topic = guess_topic(rel_path, default_topic)
                    company = guess_company(file)
                    
                    # Create normalized chunk-schema record
                    record = {
                        "chunk_id": str(uuid.uuid4()),
                        "text": content.strip(),
                        "source_type": "github",
                        "source_name": f"{repo_name}/{rel_path}".replace("\\", "/"),
                        "senior_id": senior_id,
                        "topic": topic,
                        "company": company,
                        "date_added": datetime.date.today().isoformat()
                    }
                    all_records.append(record)
                    files_extracted += 1

        print(f"  -> Extracted {files_extracted} markdown/text files from {repo_name}.")

    # Save all extracted records to JSON Lines file
    os.makedirs(config.PROCESSED_DIR, exist_ok=True)
    github_chunks_raw_path = os.path.join(config.PROCESSED_DIR, "github_chunks_raw.jsonl")
    
    with open(github_chunks_raw_path, "w", encoding="utf-8") as f:
        for record in all_records:
            f.write(json.dumps(record) + "\n")

    print("\n================ GitHub Ingestion Summary ================")
    print(f"Repositories processed : {repos_processed}/{len(repos)}")
    print(f"Total documents extracted: {len(all_records)}")
    print(f"Saved output file      : {github_chunks_raw_path}")
    print("==========================================================\n")

if __name__ == "__main__":
    ingest_github_repos()
