import os
import json
import time
from pathlib import Path
import pypdf
import ollama
from tavily import TavilyClient
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Configuration
RESUME_PATH = "workspace/Resume2026.pdf" #path to your resume
OUTPUT_PATH = "workspace/matched_jobs.md" #path where you want matched job file
SEEN_JOBS_PATH = "" #path where you want jobs already seen file
TAVILY_API_KEY = "" #enter your tavily key for websearching
TARGET_MATCHES = 5  # Target number of qualified matches (Score >= 5.0) per run

# Email Alert Configuration (Fill these in with your details)
ENABLE_EMAIL_ALERTS = True
SMTP_SERVER = "smtp.office365.com"  # Use "smtp.office365.com" if using Outlook
SMTP_PORT = 587
SENDER_EMAIL = "" # your email
APP_PASSWORD = "" #your email password
RECEIVER_EMAIL = "" #your email

# Model Assignments
MODEL_SOURCER = "qwen3.5:9b"      # Step 1: Sourcing & Initial Draft
MODEL_ANALYST = "phi4:14b"      # Step 2: Deep Match & Structuring
MODEL_AUDITOR = "gemma4:e4b"    # Step 3: Quality Control & Polish

SEARCH_QUERIES = [ #!!Enter job titles here for searching job boards!!
    # --- Greenhouse (Direct-Hire Tech) ---
    "remote data engineer site:boards.greenhouse.io -staffing -recruiting -consulting",
    "remote etl developer site:boards.greenhouse.io -staffing -recruiting -consulting",
    "remote analytics engineer site:boards.greenhouse.io -staffing -recruiting -consulting",
    "remote data platform engineer site:boards.greenhouse.io -staffing -recruiting -consulting",

    # --- Lever (Modern Startups & Scale-ups) ---
    "remote data engineer site:jobs.lever.co -staffing -recruiting",
    "remote etl developer site:jobs.lever.co -staffing -recruiting",
    "remote analytics engineer site:jobs.lever.co -staffing -recruiting",
    "remote bi engineer site:jobs.lever.co -staffing -recruiting",

    # --- Ashby (Fast-Growing Tech & AI Companies) ---
    "remote data engineer site:ashbyhq.com -staffing -recruiting",
    "remote analytics engineer site:ashbyhq.com -staffing -recruiting",
    "remote data platform engineer site:ashbyhq.com -staffing -recruiting",

    # --- SmartRecruiters (Enterprise Direct-Hires) ---
    "remote data engineer site:jobs.smartrecruiters.com -staffing -recruiting",
    "remote analytics engineer site:jobs.smartrecruiters.com -staffing -recruiting",

    # --- Workday (Major Corporate Direct-Hires) ---
    "remote data engineer site:myworkdayjobs.com -staffing -recruiting",
    "remote analytics engineer site:myworkdayjobs.com -staffing -recruiting",
]

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
    os.makedirs("workspace", exist_ok=True)
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
    """Fetches real job listings and filters out anything already seen."""
    all_jobs = []
    print("[Info] Sourcing wide-net job listings via Tavily API...")
    
    for query in SEARCH_QUERIES:
        try:
            print(f"-> Query: {query}")
            response = client.search(query=query, max_results=5, search_depth="advanced")
            for r in response.get("results", []):
                url = r.get("url", "")
                if url and url not in seen_urls:
                    all_jobs.append({
                        "title": r.get("title", "Unknown Title"),
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
    """Sends an email alert containing the job matches report."""
    if not ENABLE_EMAIL_ALERTS:
        return
        
    print("[Info] Sending email notification...")
    try:
        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = RECEIVER_EMAIL
        msg['Subject'] = f"?? Job Pipeline Complete: {job_count} New Matches Ready"

        body = f"Your daily multi-LLM job search pipeline has finished running.\n\nHere are your top qualified matches:\n\n{report_content}"
        msg.attach(MIMEText(body, 'plain'))

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SENDER_EMAIL, APP_PASSWORD)
            server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, msg.as_string())
            
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
    print(f"[Info] Scanning listings until we secure {TARGET_MATCHES} qualified matches (Score >= 5.0)...")

    for i, job in enumerate(raw_jobs, 1):
        if len(evaluated_jobs) >= TARGET_MATCHES:
            print(f"\n[Info] Target of {TARGET_MATCHES} qualified matches reached. Stopping early.")
            break

        print(f"\nProcessing Job {i}/{len(raw_jobs)}: {job['title']}")

        # --- STEP 1: QWEN ---
        qwen_res = ollama.chat(model=MODEL_SOURCER, messages=[
            {'role': 'system', 'content': 'You are an expert technical recruiter sourcing direct-hire data engineering roles. REJECT CRITERIA: If the job posting is from a third-party staffing agency, headhunter, or IT consulting contractor firm (e.g., TEKsystems, Robert Half, Apex, Dice-posters), output a SCORE of 0 and state that it is a staffing agency.'},
            {'role': 'user', 'content': f"Analyze this job posting against the candidate's resume.\nRESUME: {resume_text}\nJOB TITLE: {job['title']}\nJOB: {job['content']}\nProvide a preliminary fit assessment."}
        ])['message']['content']

        # --- STEP 2: PHI-4 ---
        phi_res = ollama.chat(model=MODEL_ANALYST, messages=[
            {'role': 'system', 'content': 'You are a strict data engineering hiring manager. Be objective.'},
            {'role': 'user', 'content': f"Take Qwen's draft, assign a score (1-10), and draft a 2-sentence rationale.\nDRAFT: {qwen_res}\nFormat strictly:\nSCORE: [1-10]\nRATIONALE: [2-sentence explanation]"}
        ])['message']['content']

        # --- STEP 3: GEMMA ---
        gemma_res = ollama.chat(model=MODEL_AUDITOR, messages=[
            {'role': 'system', 'content': 'You are an elite QA auditor for recruitment intelligence.'},
            {'role': 'user', 'content': r"Review and clean this evaluation:\n" + phi_res + r"\nReturn strictly:\nSCORE: [Number]\nRATIONALE: [Polished text]"}
        ])['message']['content']

        score = 5.0
        rationale = gemma_res
        for line in gemma_res.split('\n'):
            if line.startswith("SCORE:"):
                try:
                    score = float(line.replace("SCORE:", "").strip().split()[0])
                except:
                    pass

        seen_urls.add(job['url'])

        if score >= 5.0:
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
        print("\n[Info] No qualified matches found meeting the threshold in this batch.")
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

    os.makedirs("workspace", exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write("# Daily Verified Job Matches Report\n\n")
        f.write(f"Generated on {time.strftime('%Y-%m-%d')} for qualified listings (Score >= 5.0).\n\n")
        f.write("| Score | Job Title | Application Link | Multi-LLM Rationale |\n")
        f.write("| :---: | :--- | :--- | :--- |\n")
        
        for job in evaluated_jobs:
            f.write(f"| **{job['score']} / 10** | {job['title']} | [Apply Here]({job['url']}) | {job['evaluation'].replace(chr(10), ' ')} |\n")

    print(f"\n[Success] Pipeline complete! {len(evaluated_jobs)} qualified matches added to {OUTPUT_PATH}.")
    
    # Send Email Notification
    send_email_notification(len(evaluated_jobs), report_text)

if __name__ == "__main__":
    main()