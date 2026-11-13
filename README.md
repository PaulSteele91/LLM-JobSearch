# Multi-LLM Automated Job Search & Recruitment Intelligence Pipeline

An automated recruitment intelligence pipeline designed to source direct-hire data engineering and analytics roles across 13+ ATS platforms, evaluate them against a candidate resume using a local 3-tier LLM consensus committee, and deliver structured match reports directly to markdown and email.

---

## Key Features & Architecture

### 1. 3-Tier Local LLM Consensus Committee
The pipeline orchestrates three distinct local open-weights models running via **Ollama**, right-sized to execute 100% within VRAM on a 12GB GPU (e.g. NVIDIA RTX 4070 / 5070):

* **Tier 1 — Sourcer (`qwen2.5:14b`):** 
  * Ingests candidate resume and raw job listings.
  * Filters out third-party staffing agencies, non-remote roles, and geographic mismatches.
  * **Short-Circuit Optimization:** If the sourcer assigns a score of `0.0`, the pipeline discards the listing immediately, saving ~70% inference time by skipping downstream models.
* **Tier 2 — Analyst (`phi4:14b`):**
  * Acts as a conservative hiring manager.
  * Evaluates core technical depth, distributed data systems experience, and role seniority.
  * Produces structured technical rationale.
* **Tier 3 — Quality Auditor (`gemma4:12b`):**
  * Cross-references the evaluation against the candidate resume and job posting.
  * Calibrates the final score (0.0 to 10.0), verifies company HQ / remote distribution, and ensures clean formatting.

### 2. Broad Multi-ATS Sourcing (13 Platforms)
Directly targets career pages and postings hosted on primary Applicant Tracking Systems:
* **Greenhouse** (`greenhouse.io`)
* **Lever** (`jobs.lever.co`)
* **Ashby** (`ashbyhq.com`)
* **Rippling** (`jobs.rippling.com`)
* **Workable** (`apply.workable.com`)
* **SmartRecruiters** (`jobs.smartrecruiters.com`)
* **Workday** (`myworkdayjobs.com`)
* **Breezy HR** (`breezy.hr`)
* **iCIMS** (`icims.com`)
* **Jobvite** (`jobs.jobvite.com`)
* **BambooHR** (`bamboohr.com`)
* **Recruitee** (`recruitee.com`)
* **Pinpoint** (`pinpointhq.com`)

### 3. Edge-Case Filtering & Anti-Scraping Resilience
* **Workday Ghost-Job & Empty-Shell Detection:** Detects expired single-page applications (SPAs) where HTTP 200 is returned but the requisition has closed (inspecting `<title></title>` and blank OpenGraph metadata).
* **Soft-404 Filtering:** Rejects postings that contain expired, filled, or deleted notice phrasing.
* **Title & Aggregator Guardrails:** Regex filtering to discard category directories, "browse jobs" hubs, and senior/lead/management postings.
* **Dual-Pass Query Slicing:** Splits queries into focused batches (e.g. Core Distributed Data vs. Analytics / ETL) to maximize search engine indexing limits.

### 4. State Management & Alerts
* **Persistent Disk Cache (`seen_jobs.json`):** Tracks processed URLs across runs to guarantee zero duplicate evaluations.
* **Clean Markdown Reports (`matched_jobs.md`):** Generates structured comparison tables with scores, locations, direct apply links, and multi-LLM rationales.
* **Automated Email Dispatch:** Optional email summaries dispatched via the **Resend API**.

---

## Getting Started

### 1. Prerequisites
* Python 3.9+
* [Ollama](https://ollama.ai) installed with required models pulled:
  ```bash
  ollama pull qwen2.5:14b
  ollama pull phi4:14b
  ollama pull gemma4:12b
  ```
* A [Tavily API](https://tavily.com) key for ATS web search indexing.
* *(Optional)* A [Resend API](https://resend.com) key for email alerts.

### 2. Installation
```bash
git clone https://github.com/PaulSteele91/LLM-JobSearch.git
cd LLM-JobSearch
pip install -r requirements.txt
```
*(Or install directly: `pip install pypdf ollama tavily-python resend requests`)*

### 3. Configuration
Copy the template configuration file:
```bash
cp .env.example .env
```
Fill in your credentials and preferences in `.env` (or pass them as environment variables):
```ini
TAVILY_API_KEY=tvly-your-key-here
RESUME_PATH=./path_to_your_resume.pdf
SCORE_THRESHOLD=7.0
TARGET_MATCHES=5
```

### 4. Execution
```bash
python LLM-JobSearch.py
```

The script will search across target ATS boards, run the consensus evaluation committee, write qualified matches to `matched_jobs.md`, and (optionally) send an email alert.

---

## License
MIT License. Feel free to fork, adapt, and customize for your own job search workflows.
