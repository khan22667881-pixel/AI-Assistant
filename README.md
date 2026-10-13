# AI-Assistant
# 📄 ATS Resume Checker

Upload a resume (PDF, DOCX or TXT) and get an **estimated ATS score** with prioritized, concrete improvements. Optionally paste a job description for a targeted keyword match. Built with Streamlit and Google Gemini Flash.

> The score is an AI-based estimate. No real ATS publishes a universal score, so use it as guidance, not a guarantee.

## Features
- Overall score (0-100) from five weighted categories: keywords, content impact, formatting, structure, readability
- Prioritized improvements (High / Medium / Low) with rewrite suggestions
- Found vs. missing keywords, optionally against a job description
- Formatting red flags and scanned-PDF detection
- Downloadable Markdown report

## Run locally
```bash
git clone https://github.com/<your-username>/ats-resume-checker.git
cd ats-resume-checker
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Get a free API key at https://aistudio.google.com/apikey, then either:

```bash
export GEMINI_API_KEY="your-key"     # Windows PowerShell: $env:GEMINI_API_KEY="your-key"
streamlit run app.py
```
or create `.streamlit/secrets.toml`:
```toml
GEMINI_API_KEY = "your-key"
# GEMINI_MODEL = "gemini-2.5-flash"   # optional override
```
If no key is configured, the app asks for one in the sidebar.

## Deploy on Streamlit Community Cloud
1. Push this repo to GitHub (never commit your key; `.gitignore` already excludes secrets).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app**, pick the repo, branch `main`, main file `app.py`.
4. Open **Advanced settings → Secrets** and paste: `GEMINI_API_KEY = "your-key"`
5. Click **Deploy**.

## Project files
- `app.py` – the Streamlit app
- `requirements.txt` – dependencies

## Privacy
Resume text is sent to the Gemini API for analysis and is not stored by this app.
