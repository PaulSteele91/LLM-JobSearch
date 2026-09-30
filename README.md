# Multi-LLM Automated Job Search Pipeline

An automated recruitment intelligence pipeline designed to source direct-hire data engineering roles across multiple ATS platforms (Greenhouse, Lever, Ashby, SmartRecruiters, Workday), evaluate them using a 3-tier local LLM committee, and deliver curated matches directly to email.

## Architecture & Model Stack
- **Sourcing:** `qwen3.5:9b` (Wide-net extraction & corporate red-flag filtering)
- **Scoring & Logic:** `phi4:14b` (Rigorous objective hiring-manager evaluation)
- **QA Auditing:** `gemma4:e4b` (Final formatting and markdown enforcement)
- **Search Engine:** Tavily API
- **State Management:** JSON-based persistence to prevent duplicate processing.

## Environment & Compatibility
- Developed and tested on **Linux / Unix** (compatible with macOS and Windows Git Bash).
- Requires Python 3.8+ and local Ollama installation for multi-LLM orchestration.

## Quick Start
1. Clone the repository and install dependencies:
  pip install pypdf ollama tavily-python

2.Configure your API keys, resume path, and email credentials in LLM_JobSearch.py.

3.Run the pipeline:
  python LLM_JobSearch.py
