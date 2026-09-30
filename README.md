# Multi-LLM Automated Job Search Pipeline

An automated recruitment intelligence pipeline designed to source direct-hire data engineering roles across multiple ATS platforms (Greenhouse, Lever, Ashby, SmartRecruiters, Workday), evaluate them using a 3-tier local LLM committee, and deliver curated matches directly via email.

## Architecture & Model Stack
- **Sourcing:** `qwen3.5:9b` (Wide-net extraction & corporate red-flag filtering)
- **Scoring & Logic:** `phi4:14b` (Rigorous objective hiring-manager evaluation)
- **QA Auditing:** `gemma4:e4b` (Final formatting and markdown enforcement)
- **Search Engine:** Tavily API (Used via `tavily-python` for high-precision web discovery and deep searching)
- **State Management:** JSON-based persistence to prevent duplicate processing.

## Notification & Delivery
- **Email Delivery:** Dispatches clean, formatted HTML job matches straight to your inbox via the **Resend API** (`resend`).
- **Configuration:** Set your Tavily and Resend API keys, resume path, and recipient email inside `LLM_JobSearch.py` or via environment variables (`RESEND_API_KEY`, `TAVILY_API_KEY`).

## Environment & Compatibility
- Developed and tested on **Linux / Unix** (compatible with macOS and Windows Git Bash).
- Requires Python 3.8+ and local Ollama installation for multi-LLM orchestration.

## Quick Start
1. Clone the repository and install dependencies:
   ```bash
   pip install pypdf ollama tavily-python resend