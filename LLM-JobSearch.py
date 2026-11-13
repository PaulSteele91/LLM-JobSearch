import os
import json
import time
from pathlib import Path
import re
import requests
import pypdf
import ollama
from tavily import TavilyClient
import resend

# ==========================================
# Configuration & Environment Variables
# ==========================================
# Load from environment variables if present, or fallback to sensible defaults
RESUME_PATH = os.getenv("RESUME_PATH", "./sample_resume.pdf")
OUTPUT_PATH = os.getenv("OUTPUT_PATH", "./matched_jobs.md")
SEEN_JOBS_PATH = os.getenv("SEEN_JOBS_PATH", "./seen_jobs.json")

# API Keys (Set via environment variables or paste your keys here for local testing)
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
RESEND_API_KEY = os.getenv("RESEND_API_KEY", "")

# Search & Evaluation Thresholds
TARGET_MATCHES = int(os.getenv("TARGET_MATCHES", "5"))       # Target number of qualified matches per run
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "7.0"))  # Minimum match score required to accept (0.0 - 10.0)

# Email Notification Configuration
ENABLE_EMAIL_ALERTS = os.getenv("ENABLE_EMAIL_ALERTS", "False").lower() in ("true", "1", "yes")
RECEIVER_EMAIL = os.getenv("RECEIVER_EMAIL", "")

# Geographic & Remote Policy Preferences
# Specify preferred hubs or "Remote-First / Distributed"
APPROVED_LOCATIONS = os.getenv(
    "APPROVED_LOCATIONS",
    "Remote-First / Distributed, Coastal California, Greater Denver (CO), New York City Metro, Massachusetts, or Florida"
)

# Model Assignments (Tuned for 12GB VRAM local Ollama execution)
MODEL_SOURCER = os.getenv("MODEL_SOURCER", "qwen2.5:14b")   # Step 1: Sourcing, Broad Screening & Early Discard
MODEL_ANALYST = os.getenv("MODEL_ANALYST", "phi4:14b")       # Step 2: Deep Technical Match & Structuring
MODEL_AUDITOR = os.getenv("MODEL_AUDITOR", "gemma4:12b")     # Step 3: Quality Control, Scoring & Rationale

# Split search queries into focused passes for higher indexing coverage across ATS boards
SEARCH_QUERIES = [
    'remote "United States" ("data engineer" OR "big data engineer" OR "data platform engineer") -senior -sr -lead -staff -principal -manager -director -staffing -recruiting',
    'remote "United States" ("analytics engineer" OR "ETL developer" OR "data integration engineer" OR "data quality engineer") -senior -sr -lead -staff -principal -manager -director -staffing -recruiting'
]

# Supported Applicant Tracking System (ATS) domains
ATS_DOMAINS = [
    "greenhouse.io",
    "jobs.lever.co",
    "ashbyhq.com",
    "jobs.rippling.com",
    "jobs.smartrecruiters.com",
    "myworkdayjobs.com",
    "apply.workable.com",
    "breezy.hr",
    "icims.com",
    "jobs.jobvite.com",
    "bamboohr.com",
    "recruitee.com",
    "pinpointhq.com"
]


def is_valid_url(url: str) -> bool:
    """
    Checks if a URL is reachable, returns 200, and does not contain soft-404 / expired job messages.
    Includes explicit detection for Workday empty single-page application (SPA) shells.
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        # Use 5s timeout and stream response to avoid downloading massive payloads unnecessarily
        response = requests.get(url, headers=headers, timeout=5, allow_redirects=True, stream=True)
        
        if response.status_code != 200:
            return False
            
        # Inspect the first 100KB of response
        content_chunk = next(response.iter_content(chunk_size=102400), b"").decode("utf-8", errors="ignore")
        body_text = content_chunk.lower()
        
        # Common phrases used by ATS platforms for dead or expired jobs
        soft_404_phrases = [
            "page you are looking for doesn't exist",
            "page you are looking for does not exist",
            "no longer available",
            "job expired",
            "position has been filled",
            "this job posting is no longer active",
            "job is closed",
            "we can't find that page"
        ]
        
        # Workday specific dead link check:
        # When a Workday job is deleted/filled, Workday leaves <title></title> empty and strips content from og:title
        if "myworkdayjobs.com" in url.lower():
            if re.search(r"<title>\s*</title>", content_chunk, re.IGNORECASE) or re.search(r'<meta\s+name="title"\s+property="og:title"\s*>', content_chunk, re.IGNORECASE):
                print(f"-> Discarded (Workday Inactive / Blank Job Shell Detected): {url}")
                return False

        if any(phrase in body_text for phrase in soft_404_phrases):
            print(f"-> Discarded (Soft-404 / Expired Job Detected): {url}")
            return False
            
        return True
    except Exception:
        return False


def is_valid_job_title(title: str) -> bool:
    """Pre-filters aggregator landing pages, career catalogs, and senior-level management roles."""
    title_lower = title.lower()
    
    # Reject plural "jobs" or "careers" (catches aggregator/category landing pages)
    if re.search(r"\b(jobs|careers)\b", title_lower):
        return False
        
    # Reject obvious search/browse directory phrases
    aggregator_phrases = ["search", "browse", "careers at", "all openings", "open positions", "opportunities in"]
    if any(phrase in title_lower for phrase in aggregator_phrases):
        return False

    # Reject senior / lead / management titles
    if re.search(r"\b(senior|sr\.?|lead|staff|principal|manager|director|head of|vp)\b", title_lower):
        return False
        
    return True


def load_seen_jobs() -> set:
    """Loads previously processed job URLs from persistent disk storage."""
    if os.path.exists(SEEN_JOBS_PATH):
        try:
            with open(SEEN_JOBS_PATH, "r", encoding="utf-8") as f:
                return set(json.load(f))
        except Exception:
            return set()
    return set()


def save_seen_jobs(seen_set: set):
    """Saves updated set of seen job URLs to persistent disk storage."""
    with open(SEEN_JOBS_PATH, "w", encoding="utf-8") as f:
        json.dump(list(seen_set), f, indent=4)


def read_resume(path_str: str) -> str:
    """Extracts text content from a PDF or plain text resume file."""
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Resume file not found at '{path_str}'. Please verify RESUME_PATH.")
        
    if path.suffix.lower() == ".pdf":
        reader = pypdf.PdfReader(str(path))
        text_parts = [page.extract_text() for page in reader.pages if page.extract_text()]
        return "\n".join(text_parts).strip()
    else:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()


def fetch_job_listings(client: TavilyClient, seen_urls: set) -> list:
    """Sources brand-new job listings across ATS platforms via Tavily API and applies pre-filtering."""
    all_jobs = []
    print("[Info] Sourcing wide-net job listings via Tavily API...")
    
    blacklist_titles = ["page_title", "greenhouse", "lever", "ashby", "workday", "smartrecruiters", "sign in", "log in"]

    for domain in ATS_DOMAINS:
        for query in SEARCH_QUERIES:
            try:
                response = client.search(
                    query=query,
                    max_results=20,
                    search_depth="advanced",
                    include_domains=[domain],
                    time_range="month",
                    include_raw_content=True
                )
                for r in response.get("results", []):
                    url = r.get("url", "")
                    title = r.get("title", "Unknown Title").strip()
                    
                    # Filter junk search title artifacts
                    if any(bad in title.lower() for bad in blacklist_titles) and len(title.split()) < 3:
                        continue
                        
                    if not is_valid_job_title(title):
                        continue
                        
                    if url and url not in seen_urls:
                        print(f"-> Verifying URL active status: {url}")
                        if not is_valid_url(url):
                            print(f"-> Discarded broken/dead link: {url}")
                            continue
                            
                        raw = r.get("raw_content") or ""
                        snippet = r.get("content", "")
                        job_text = raw if len(raw) > len(snippet) else snippet
                        
                        # Pre-filter explicit international hubs without US remote availability
                        intl_blacklist = [
                            r"\b(india|bangalore|bengaluru|hyderabad|pune|mumbai|delhi|noida)\b",
                            r"\b(united kingdom|london|emea|apac|philippines|manila|poland|germany|berlin)\b"
                        ]
                        content_lower = (title + " " + job_text[:2000]).lower()
                        if any(re.search(pat, content_lower) for pat in intl_blacklist):
                            if not re.search(r"\b(remote[, -]+united states|remote[, -]+us|us remote|united states remote)\b", content_lower):
                                print(f"-> Filtered out international / non-US posting: {title}")
                                continue
                            
                        all_jobs.append({
                            "title": title,
                            "url": url,
                            "content": job_text[:10000]
                        })
                        
            except Exception as e:
                print(f"[Warning] Query failed for domain '{domain}': {e}")
            
    # Deduplicate within current batch
    unique_jobs = []
    batch_seen = set()
    for job in all_jobs:
        if job['url'] not in batch_seen:
            batch_seen.add(job['url'])
            unique_jobs.append(job)
            
    print(f"[Info] Found {len(unique_jobs)} brand-new unique job listings to evaluate.")
    return unique_jobs


def send_email_notification(job_count: int, report_content: str):
    """Sends an email alert containing the job matches report via Resend."""
    if not ENABLE_EMAIL_ALERTS or not RESEND_API_KEY or not RECEIVER_EMAIL:
        print("[Info] Email alerts disabled or configuration missing (RESEND_API_KEY / RECEIVER_EMAIL).")
        return
        
    print("[Info] Sending email notification via Resend...")
    try:
        resend.api_key = RESEND_API_KEY
        resend.Emails.send({
            "from": "onboarding@resend.dev",
            "to": RECEIVER_EMAIL,
            "subject": f"Job Pipeline Complete: {job_count} High-Fit Matches Ready",
            "text": f"Your daily multi-LLM job search pipeline has finished running.\n\nHere are your top qualified matches (Score >= {SCORE_THRESHOLD}):\n\n{report_content}"
        })
        print("[Success] Email notification sent successfully!")
    except Exception as e:
        print(f"[Warning] Failed to send email notification: {e}")


def main():
    print("=== Starting Multi-LLM Job Intelligence Pipeline ===")
    
    if not TAVILY_API_KEY:
        print("[Error] TAVILY_API_KEY is not set. Please set it in your environment or configuration.")
        return
        
    client = TavilyClient(api_key=TAVILY_API_KEY)
    seen_urls = load_seen_jobs()
    
    try:
        resume_text = read_resume(RESUME_PATH)
    except Exception as e:
        print(f"[Error] {e}")
        return

    raw_jobs = fetch_job_listings(client, seen_urls)
    if not raw_jobs:
        print("[Info] No new job listings found today. Check back later!")
        return

    evaluated_jobs = []
    print(f"[Info] Scanning listings until securing {TARGET_MATCHES} qualified matches (Score >= {SCORE_THRESHOLD})...")

    try:
        for i, job in enumerate(raw_jobs, 1):
            if len(evaluated_jobs) >= TARGET_MATCHES:
                print(f"\n[Info] Target of {TARGET_MATCHES} qualified matches reached. Stopping early.")
                break

            print(f"\nProcessing Job {i}/{len(raw_jobs)}: {job['title']}")

            # --- STEP 1: SOURCER (Qwen) ---
            qwen_res = ollama.chat(model=MODEL_SOURCER, messages=[
                {'role': 'system', 'content': (
                    'You are an objective technical sourcer screening direct-hire data engineering listings. '
                    'Be realistic, conservative, and strict regarding experience, tenure, and location requirements.\n\n'
                    'REJECTION CRITERIA (Assign SCORE: 0):\n'
                    '1. NOT 100% REMOTE: If the role is hybrid, onsite, requires weekly in-office presence, or mandates relocation, reject it.\n'
                    f'2. GEOGRAPHIC / RTO MISMATCH: The candidate accepts 100% remote positions that are remote-first/distributed or aligned with approved locations: {APPROVED_LOCATIONS}. Reject roles anchored to single regional campuses outside these areas.\n'
                    '3. THIRD-PARTY STAFFING: If from a third-party agency, contract firm, or headhunter, reject it.\n'
                    '4. INCOMPLETE: If not a specific job posting or too incomplete to evaluate, reject it.\n\n'
                    'Format strictly:\n'
                    'SCORE: [0-10]\n'
                    'LOCATION: [Company HQ / Location OR "Remote-First / Distributed"]\n'
                    'PRELIMINARY: [Explanation]'
                )},
                {'role': 'user', 'content': f"Analyze this job posting against the candidate's resume.\nRESUME:\n{resume_text}\n\nJOB TITLE: {job['title']}\nJOB POSTING:\n{job['content']}\n\nProvide preliminary assessment."}
            ],
            options={"num_ctx": 8192}
            )['message']['content']

            # Extract Sourcer score
            qwen_score_matches = re.findall(r"(?i)score\s*[:*]*\s*\[?([0-9]*\.?[0-9]+)\]?", qwen_res)
            qwen_score = float(qwen_score_matches[-1]) if qwen_score_matches else 5.0
            seen_urls.add(job['url'])

            # Early Discard: Short-circuit disqualified or irrelevant roles to save inference time
            if qwen_score == 0.0:
                print("-> Discarded by Sourcer (Score 0.0: Disqualified / Ineligible). Skipping Analyst & Auditor.")
                time.sleep(0.2)
                continue

            # --- STEP 2: ANALYST (Phi-4) ---
            phi_res = ollama.chat(model=MODEL_ANALYST, messages=[
                {'role': 'system', 'content': (
                    'You are a strict data engineering hiring manager. Be objective, realistic, and conservative. '
                    f'Verify technical fit, core tooling alignment, and remote flexibility ({APPROVED_LOCATIONS}). '
                    'If the job text is invalid, hybrid, or poor fit, assign SCORE: 0.'
                )},
                {'role': 'user', 'content': (
                        f"Evaluate this candidate's resume against the job posting and preliminary sourcer draft.\n\n"
                        f"RESUME:\n{resume_text}\n\n"
                        f"JOB TITLE: {job['title']}\n\n"
                        f"JOB POSTING:\n{job['content']}\n\n"
                        f"PRELIMINARY DRAFT:\n{qwen_res}\n\n"
                        f"Assign an objective score (0-10), verify company distribution/HQ, and write a 2-sentence rationale.\n"
                        f"Format strictly:\nSCORE: [1-10]\nLOCATION: [HQ location or 'Remote-First / Distributed']\nRATIONALE: [2-sentence explanation]"
                    )
                }
            ],
            options={"num_ctx": 8192}
            )['message']['content']

            # --- STEP 3: AUDITOR (Gemma) ---
            gemma_res = ollama.chat(model=MODEL_AUDITOR, messages=[
                {'role': 'system', 'content': (
                    "You are a strict quality auditor for recruitment intelligence. Verify technical accuracy, tenure, and remote constraints.\n\n"
                    "HARD REJECTION CRITERIA (Assign SCORE: 0.0):\n"
                    "- NOT 100% REMOTE: Reject if hybrid, on-site, or requiring periodic office visits.\n"
                    f"- LOCATION COMPLIANCE: Must be Remote-First / Distributed or located in approved regions: {APPROVED_LOCATIONS}.\n"
                    "- Discard if incomplete or invalid.\n\n"
                    "EVALUATION CRITERIA:\n"
                    "1. Base technical match score (0.0 - 10.0).\n"
                    "2. Cap score at 10.0.\n\n"
                    "REQUIRED OUTPUT FORMAT:\n"
                    "LOCATION: [Company HQ / Location OR 'Remote-First / Distributed']\n"
                    "SCORE: [Number 0.0-10.0]\n"
                    "RATIONALE: [2-sentence explanation]"
                )},
                {'role': 'user', 'content': f"Cross-reference this evaluation against the resume and job description.\n\nRESUME:\n{resume_text}\n\nJOB TITLE: {job['title']}\n\nJOB POSTING:\n{job['content']}\n\nPREVIOUS EVAL:\n{phi_res}\n\nReturn strictly:\nLOCATION: [Location]\nSCORE: [Number]\nRATIONALE: [Polished text]"}
            ],
            options={"num_ctx": 8192}
            )['message']['content']

            score = 0.0
            score_matches = re.findall(r"(?i)score\s*[:*]*\s*\[?([0-9]*\.?[0-9]+)\]?", gemma_res)
            if score_matches:
                try:
                    score = float(score_matches[-1])
                except ValueError:
                    score = 0.0

            # Extract Location
            location_match = re.search(r"(?i)location\s*[:*]*\s*([^\n\r]+)", gemma_res)
            location_str = location_match.group(1).strip() if location_match else "Remote-First / Distributed"

            # Extract Rationale Cleanly
            rationale_match = re.search(r"(?i)rationale\s*[:*]*\s*(.*)", gemma_res, re.DOTALL)
            if rationale_match:
                clean_rationale = rationale_match.group(1).strip()
            else:
                clean_rationale = re.sub(r"(?im)^\s*(score|location)\s*[:*].*$", "", gemma_res).strip()

            if score >= SCORE_THRESHOLD:
                evaluated_jobs.append({
                    "title": job['title'],
                    "location": location_str,
                    "url": job['url'],
                    "score": score,
                    "evaluation": clean_rationale
                })
                print(f"-> Match Accepted! Score: {score}/10 | Location: {location_str}")
            else:
                print(f"-> Discarded (Score {score}/10 is below threshold).")

            time.sleep(0.5)

    finally:
        save_seen_jobs(seen_urls)

    if not evaluated_jobs:
        print(f"\n[Info] No qualified matches found meeting threshold ({SCORE_THRESHOLD}) in this batch.")
        return

    evaluated_jobs.sort(key=lambda x: x['score'], reverse=True)

    # Build Report Content
    report_lines = []
    for job in evaluated_jobs:
        safe_title = job['title'].replace('|', '-').strip()
        safe_loc = job.get('location', 'Remote').replace('|', '-').strip()
        safe_eval = job['evaluation'].replace('|', '-').replace('\n', ' ').strip()
        report_lines.append(f"Score: {job['score']} / 10")
        report_lines.append(f"Title: {safe_title}")
        report_lines.append(f"Location/HQ: {safe_loc}")
        report_lines.append(f"Link: {job['url']}")
        report_lines.append(f"Rationale: {safe_eval}")
        report_lines.append("-" * 40)
    
    report_text = "\n".join(report_lines)

    # Write Markdown Report Table
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write("# Daily Verified Job Matches Report\n\n")
        f.write(f"Generated on {time.strftime('%Y-%m-%d')} for qualified listings (Score >= {SCORE_THRESHOLD}).\n\n")
        f.write("| Score | Job Title | Location / HQ | Application Link | Multi-LLM Rationale |\n")
        f.write("| :---: | :--- | :--- | :--- | :--- |\n")
        
        for job in evaluated_jobs:
            safe_title = job['title'].replace('|', '-').strip()
            safe_loc = job.get('location', 'Remote').replace('|', '-').strip()
            safe_eval = job['evaluation'].replace('|', '-').replace('\n', ' ').strip()
            f.write(f"| **{job['score']} / 10** | {safe_title} | {safe_loc} | [Apply Here]({job['url']}) | {safe_eval} |\n")

    print(f"\n[Success] Pipeline complete! {len(evaluated_jobs)} qualified matches added to {OUTPUT_PATH}.")
    
    # Send Email Notification
    send_email_notification(len(evaluated_jobs), report_text)


if __name__ == "__main__":
    main()
