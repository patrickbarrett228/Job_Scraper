"""
Multi-Resume Gemini Evaluator Engine (100% Free Tier)
Compares scraped job postings against all RESUME_* secrets set in GitHub Actions.
"""

import json
import os
import time
from urllib.request import Request, urlopen

def load_resumes() -> dict[str, str]:
    """Load candidate resumes from GitHub Secrets / environment variables."""
    resumes = {}
    for key, val in os.environ.items():
        if key.startswith("RESUME_") and val.strip():
            label = key.removeprefix("RESUME_").replace("_", " ").title()
            resumes[label] = val.strip()

    if not resumes:
        print("  ⚠️ No RESUME_* environment variables found.")

    return resumes


def evaluate_job_with_gemini(job: dict, resumes: dict[str, str]) -> dict:
    """Uses Google's free Gemini API to evaluate a job against all provided resumes."""
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        return {"best_resume": "N/A", "fit_score": 0, "reasons": ["GEMINI_API_KEY secret missing."]}

    prompt = f"""
    You are an expert executive tech recruiter and resume evaluator.
    Compare the following Job Posting against all provided candidate resumes.

    Job Title: {job.get('title')}
    Company: {job.get('company')}
    Location: {job.get('location')}

    Candidate Resumes:
    {json.dumps(resumes, indent=2)}

    Task:
    1. Identify which resume is the BEST fit for this job.
    2. Calculate a fit_score (integer from 0 to 100).
    3. Provide 2-3 specific bullet points explaining the match or gaps.

    Respond STRICTLY in JSON format:
    {{
        "best_resume": "<matching_resume_key>",
        "fit_score": <0-100 integer>,
        "reasons": ["<reason 1>", "<reason 2>"]
    }}
    """

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={gemini_key}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"response_mime_type": "application/json"}
    }

    req = Request(url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})

    try:
        with urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            result_text = data['candidates'][0]['content']['parts'][0]['text']
            return json.loads(result_text)
    except Exception as e:
        print(f"  ⚠️️ Gemini evaluation error for {job.get('title')}: {e}")
        return {"best_resume": "Error", "fit_score": 0, "reasons": [str(e)]}


def process_and_score_jobs(jobs: list[dict], min_fit: int = 75) -> list[dict]:
    """Scrapes jobs, scores them against resumes with Gemini, and filters top matches."""
    resumes = load_resumes()
    if not resumes:
        return jobs

    print(f"\n🤖 Running Gemini Evaluation across {len(resumes)} resume profile(s)...")

    scored_jobs = []

    for idx, job in enumerate(jobs):
        print(f"  [{idx+1}/{len(jobs)}] Scoring: {job.get('title')} at {job.get('company')}")

        # Gemini Free Tier rate limit guard: 15 calls/min -> 4 second delay
        time.sleep(4.0)

        eval_res = evaluate_job_with_gemini(job, resumes)

        job["best_resume"] = eval_res.get("best_resume", "None")
        job["fit_score"] = eval_res.get("fit_score", 0)
        job["reasons"] = eval_res.get("reasons", [])

        if job["fit_score"] >= min_fit:
            scored_jobs.append(job)

    scored_jobs.sort(key=lambda x: x["fit_score"], reverse=True)
    return scored_jobs
