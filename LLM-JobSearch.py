import os
import json
import time
from pathlib import Path
import pypdf
import ollama
from tavily import TavilyClient
import resend
import re
import requests

# Configuration
RESUME_PATH = "" # path to your resume
OUTPUT_PATH = "" # path where you want matched job file
SEEN_JOBS_PATH = "" # path where you want jobs already seen file
TAVILY_API_KEY = "" # enter your tavily key for websearching
RESEND_API_KEY = "" # enter your resend api key
TARGET_MATCHES = 5  # Target number of qualified matches per run
SCORE_THRESHOLD = 7.0  # Minimum match score required to accept

# Email Alert Configuration
ENABLE_EMAIL_ALERTS = True
RECEIVER_EMAIL = "" # your email

# Model Assignments
MODEL_SOURCER = "qwen3.5:9b" # Step 1: Sourcing & Initial Draft
MODEL_ANALYST = "phi4:14b" # Step 2: Deep Match & Structuring
MODEL_AUDITOR = "gemma4:12b" # Step 3: Quality Control & Polish

SEARCH_QUERIES = [
    # --- Greenhouse ---
    'remote ("United States" OR "Canada") ("data engineer" OR "etl developer" OR "analytics engineer" OR "spark developer") site:boards.greenhouse.io -staffing -recruiting -consulting',

    # --- Lever ---
    'remote ("United States" OR "Canada") ("data engineer" OR "etl developer" OR "analytics engineer" OR "spark developer") site:jobs.lever.co -staffing -recruiting -consulting',

    # --- Ashby ---
    'remote ("United States" OR "Canada") ("data engineer" OR "etl developer" OR "analytics engineer" OR "spark developer") site:ashbyhq.com -staffing -recruiting -consulting',

    # --- SmartRecruiters ---
    'remote ("United States" OR "Canada") ("data engineer" OR "etl developer" OR "analytics engineer" OR "spark developer") site:jobs.smartrecruiters.com -staffing -recruiting -consulting',

    # --- Workday ---
    'remote ("United States" OR "Canada") ("data engineer" OR "etl developer" OR "analytics engineer" OR "spark developer") site:myworkdayjobs.com -staffing -recruiting -consulting',
]

def is_valid_url(url: str) -> bool:
    """Checks if a URL is reachable, returns 200, and does not contain soft-404/expired job messages."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        # Use GET instead of HEAD so we can inspect the HTML body content
        response = requests.get(url, headers=headers, timeout=10, allow_redirects=True)
        
        # If the status code isn't 200, it's definitely broken
        if response.status_code != 200:
            return False
            
        # Convert page content to lowercase for keyword checking
        body_text = response.text.lower()
        
        # Common phrases used by Workday and other ATS platforms for dead/expired jobs
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
        
        # If any soft-404 phrase appears in the body, treat it as a dead link
        if any(phrase in body_text for phrase in soft_404_phrases):
            print(f"-> Discarded (Soft-404 / Expired Job Detected): {url}")
            return False
            
        return True
    except Exception:
        return False

def is_valid_job_title(title: str) -> bool:
    title_lower = title.lower()
    
    # 1. Reject if it contains plural "jobs" (catches "11,000+ jobs", "remote data jobs", etc.)
    if re.search(r"\bjobs\b", title_lower):
        return False
        
    # 2. Reject obvious category/aggregator landing page phrasing
    aggregator_phrases = ["search", "browse", "careers at", "all openings", "open positions", "opportunities in"]
    if any(phrase in title_lower for phrase in aggregator_phrases):
        return False
        
    # 3. Ensure it reads like a specific role
    return True

def load_seen_jobs() -> set:
    """Loads previously processed job URLs from disk."""
    if os.path.exists(SEEN_JOBS_PATH):
        try:
            with open(SEEN_JOBS_PATH, "r", encoding="utf-8") as f:
                return set(json.load(f))
        except Exception:
            return set()
    return set()

def save_seen_jobs(seen_set: set):
    """Saves the updated set of seen job URLs to disk."""
    with open(SEEN_JOBS_PATH, "w", encoding="utf-8") as f:
        json.dump(list(seen_set), f, indent=4)

def read_resume(pdf_path: str) -> str:
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"Resume not found at {pdf_path}.")
    reader = pypdf.PdfReader(str(path))
    text_parts = [page.extract_text() for page in reader.pages if page.extract_text()]
    return "\n".join(text_parts).strip()

def fetch_job_listings(client: TavilyClient, seen_urls: set) -> list:
    """Fetches real job listings and filters out anything already seen or junk."""
    all_jobs = []
    print("[Info] Sourcing wide-net job listings via Tavily API...")
    
    # Titles or terms that indicate a junk/generic search result page rather than a job
    blacklist_titles = ["page_title", "greenhouse", "lever", "ashby", "workday", "smartrecruiters", "sign in", "log in"]

    for query in SEARCH_QUERIES:
        try:
            print(f"-> Query: {query}")
            response = client.search(query=query, max_results=20, search_depth="advanced")
            for r in response.get("results", []):
                url = r.get("url", "")
                title = r.get("title", "Unknown Title").strip()
                
                # Filter out junk titles or general company career homepages
                if any(bad in title.lower() for bad in blacklist_titles) and len(title.split()) < 3:
                    continue
                    
                if not is_valid_job_title(title):
                    print(f"-> Filtered out aggregator/non-job page: {title}")
                    continue
                    
                if url and url not in seen_urls:
                    print(f"-> Verifying URL active status: {url}")
                    if not is_valid_url(url):
                        print(f"-> Discarded broken/dead link: {url}")
                        continue
                        
                    all_jobs.append({
                        "title": title,
                        "url": url,
                        "content": r.get("content", "")
                    })
                    
        except Exception as e:
            print(f"[Warning] Query failed '{query}': {e}")
            
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
    if not ENABLE_EMAIL_ALERTS or not RESEND_API_KEY:
        print("[Info] Email alerts disabled or Resend API key missing.")
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
    print("=== Starting Daily Multi-LLM Job Intelligence Pipeline ===")
    
    client = TavilyClient(api_key=TAVILY_API_KEY)
    seen_urls = load_seen_jobs()
    
    try:
        resume_text = read_resume(RESUME_PATH)
    except Exception as e:
        print(f"[Error] {e}")
        return

    raw_jobs = fetch_job_listings(client, seen_urls)
    if not raw_jobs:
        print("[Info] No new job listings found today. Check back tomorrow!")
        return

    evaluated_jobs = []
    print(f"[Info] Scanning listings until we secure {TARGET_MATCHES} qualified matches (Score >= {SCORE_THRESHOLD})...")

    for i, job in enumerate(raw_jobs, 1):
        if len(evaluated_jobs) >= TARGET_MATCHES:
            print(f"\n[Info] Target of {TARGET_MATCHES} qualified matches reached. Stopping early.")
            break

        print(f"\nProcessing Job {i}/{len(raw_jobs)}: {job['title']}")

        # --- STEP 1: QWEN ---
        qwen_res = ollama.chat(model=MODEL_SOURCER, messages=[
            {'role': 'system', 'content': 'You are a strict technical recruiter sourcing direct-hire data engineering roles. Be objective and realistic, scoring should be very conservative. Make sure to consider experience and length of experience. REJECT CRITERIA: If the job posting is from a third-party staffing agency, headhunter, or IT consulting contractor firm (e.g., TEKsystems, Robert Half, Apex, Dice-posters), output a SCORE of 0 and state that it is a staffing agency.'},
            {'role': 'user', 'content': f"Analyze this job posting against the candidate's resume.\nRESUME: {resume_text}\nJOB TITLE: {job['title']}\nJOB: {job['content']}\nProvide a preliminary fit assessment."}
        ],
        options={
        "num_ctx": 8192 #context window
        },
        )['message']['content']

        # --- STEP 2: PHI-4 ---
        phi_res = ollama.chat(model=MODEL_ANALYST, messages=[
            {'role': 'system', 'content': 'You are a strict data engineering hiring manager. Be objective and realistic, scoring should be very conservative. Make sure to consider experience and length of experience.'},
            {'role': 'user', 'content': (
                    f"Evaluate this candidate's resume against the job posting and Qwen's preliminary draft.\n\n"
                    f"RESUME: {resume_text}\n\n"
                    f"JOB TITLE: {job['title']}\n\n"
                    f"JOB: {job['content']}\n\n"
                    f"QWEN'S DRAFT: {qwen_res}\n\n"
                    f"Take Qwen's draft and the resume, assign an objective score (1-10), and draft a 2-sentence rationale.\n"
                    f"Format strictly:\nSCORE: [1-10]\nRATIONALE: [2-sentence explanation]"
                )
            }
        ],
        options={
        "num_ctx": 8192  #context window
        },
        )['message']['content']

        # --- STEP 3: GEMMA ---
        gemma_res = ollama.chat(model=MODEL_AUDITOR, messages=[
            {'role': 'system', 'content': (
                "You are a strict elite QA auditor for recruitment intelligence. Be objective and realistic, scoring should be very conservative. Make sure to consider experience and length of experience\n\n"
                "HARD REJECTION CRITERIA:\n"
                "- If position is outside US/Canada or requires international residency, assign SCORE: 0.0.\n\n"
                "EVALUATION CRITERIA:\n"
                "1. Base technical match score (1.0 - 10.0).\n"
                "2. Cap final score at 10.0.\n\n"
                "REQUIRED OUTPUT FORMAT:\n"
                "SCORE: [Number 1.0-10.0]\n"
                "RATIONALE: [2-sentence explanation]"
            )},
            {'role': 'user', 'content': f"Cross-reference this evaluation against the resume and job description to ensure absolute accuracy.\n\nRESUME: {resume_text}\nJOB TITLE: {job['title']}\nJOB: {job['content']}\nPREVIOUS EVAL: {phi_res}\n\nReturn strictly:\nSCORE: [Number]\nRATIONALE: [Polished text]"}
        ],
        options={
        "num_ctx": 8192  #context window
        },
        )['message']['content']

        score = 0.0
        rationale = gemma_res
        
        # Scans the full response (ignoring thinking tokens and brackets) to find the final score
        score_matches = re.findall(r"(?i)score\s*[:*]*\s*\[?([0-9]*\.?[0-9]+)\]?", gemma_res)
        if score_matches:
            try:
                score = float(score_matches[-1])
            except ValueError:
                score = 0.0

        seen_urls.add(job['url'])

        if score >= SCORE_THRESHOLD:
            evaluated_jobs.append({
                "title": job['title'],
                "url": job['url'],
                "score": score,
                "evaluation": rationale.replace("RATIONALE:", "").strip()
            })
            print(f"-> Match Accepted! Score: {score}/10")
        else:
            print(f"-> Discarded (Score {score}/10 is below threshold).")

        time.sleep(1)

    save_seen_jobs(seen_urls)

    if not evaluated_jobs:
        print(f"\n[Info] No qualified matches found meeting the threshold ({SCORE_THRESHOLD}) in this batch.")
        return

    evaluated_jobs.sort(key=lambda x: x['score'], reverse=True)

    # Build Report Content
    report_lines = []
    for job in evaluated_jobs:
        report_lines.append(f"Score: {job['score']} / 10")
        report_lines.append(f"Title: {job['title']}")
        report_lines.append(f"Link: {job['url']}")
        report_lines.append(f"Rationale: {job['evaluation']}")
        report_lines.append("-" * 40)
    
    report_text = "\n".join(report_lines)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write("# Daily Verified Job Matches Report\n\n")
        f.write(f"Generated on {time.strftime('%Y-%m-%d')} for qualified listings (Score >= {SCORE_THRESHOLD}).\n\n")
        f.write("| Score | Job Title | Application Link | Multi-LLM Rationale |\n")
        f.write("| :---: | :--- | :--- | :--- |\n")
        
        for job in evaluated_jobs:
            f.write(f"| **{job['score']} / 10** | {job['title']} | [Apply Here]({job['url']}) | {job['evaluation'].replace(chr(10), ' ')} |\n")

    print(f"\n[Success] Pipeline complete! {len(evaluated_jobs)} qualified matches added to {OUTPUT_PATH}.")
    
    # Send Email Notification
    send_email_notification(len(evaluated_jobs), report_text)

if __name__ == "__main__":
    main()