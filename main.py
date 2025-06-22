# main.py

from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel
import requests
from bs4 import BeautifulSoup
import time
import random
import pandas as pd
import re
import spacy
from rake_nltk import Rake
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import os

# --- API Initialization ---
app = FastAPI(
    title="LinkedIn AI Job Analyzer API",
    description="An API to scrape LinkedIn job postings and perform NLP analysis.",
    version="1.0.0"
)

# --- Data Models (using Pydantic) ---
# This defines the expected input format for our API requests.
class JobRequest(BaseModel):
    title: str
    location: str
    pages_to_scrape: int = 1
    resume_text: str | None = None # Resume text is optional

# --- Security Dependency ---
# This function will check for a secret API key in the request header.
async def get_api_key(x_api_key: str = Header(None)):
    # In a real app, you'd get this from a secure place (like an environment variable)
    # For Cloud Run, we will set this as a Secret.
    expected_api_key = os.environ.get("API_KEY") 
    if not expected_api_key or x_api_key != expected_api_key:
        raise HTTPException(status_code=401, detail="Invalid or Missing API Key")
    return x_api_key

# --- Helper & NLP Functions (copied from your previous app) ---
# NOTE: These are now regular functions, not cached with Streamlit's decorators.
def load_spacy_model():
    return spacy.load("en_core_web_sm")

def parse_time_posted(time_text):
    # ... (same function as before)
    if not time_text: return None
    num_match = re.search(r'\d+', time_text)
    if not num_match: return None
    num = int(num_match.group(0))
    time_text = time_text.lower()
    if 'second' in time_text or 'minute' in time_text or 'hour' in time_text: return 0
    elif 'day' in time_text: return num * 24
    elif 'week' in time_text: return num * 7 * 24
    elif 'month' in time_text: return num * 30 * 24
    else: return None

def parse_num_applicants(applicant_text):
    # ... (same function as before)
    if not applicant_text: return None
    num_match = re.search(r'\d+', applicant_text)
    if num_match: return int(num_match.group(0))
    else: return None

def extract_keywords(_text):
    # ... (same function as before)
    if not isinstance(_text, str) or not _text.strip(): return []
    r = Rake()
    r.extract_keywords_from_text(_text)
    return r.get_ranked_phrases()[:5]

def extract_skills(_text, _nlp_model):
    # ... (same function as before)
    if not isinstance(_text, str) or not _text.strip(): return []
    doc = _nlp_model(_text)
    return list(set([ent.text for ent in doc.ents if ent.label_ in ('ORG', 'PRODUCT')]))

def analyze_sentiment(_text):
    # ... (same function as before)
    if not isinstance(_text, str) or not _text.strip(): return 0.0
    return TextBlob(_text).sentiment.polarity

def calculate_match(job_desc, resume_text):
    # ... (same function as before)
    if not all(isinstance(t, str) and t.strip() for t in [job_desc, resume_text]): return 0.0
    vectorizer = TfidfVectorizer()
    try:
        vectors = vectorizer.fit_transform([job_desc, resume_text])
        return cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    except ValueError: return 0.0

# Pre-load the spaCy model once on startup
nlp_model = load_spacy_model()

# --- Core Logic Functions (Refactored from Streamlit) ---
def run_linkedin_scraper(title: str, location: str, num_pages: int):
    # ... (This is the same scraping logic, just without st.progress, etc.)
    id_list, start, count = [], 0, 0
    while count < num_pages:
        list_url = f"https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search?keywords={title}&location={location}&start={start}"
        response = requests.get(list_url)
        if response.status_code != 200: break
        list_soup = BeautifulSoup(response.text, "html.parser")
        page_jobs = list_soup.find_all("li")
        if not page_jobs: break
        for job in page_jobs:
            job_id = job.find("div", {"class": "base-card"}).get("data-entity-urn", "").split(":")[-1]
            if job_id and job_id not in id_list: id_list.append(job_id)
        count += 1
        start += 25
        time.sleep(random.uniform(0.5, 1.5))
    if not id_list: return None
    
    job_list = []
    for job_id in id_list:
        job_url = f"https://www.linkedin.com/jobs-guest/jobs/api/jobPosting/{job_id}"
        job_response = requests.get(job_url)
        if job_response.status_code == 200:
            job_soup = BeautifulSoup(job_response.text, "html.parser")
            time_posted_raw, num_applicants_raw, salary_raw = None, None, None
            try: time_posted_raw = job_soup.find("span", {"class": "posted-time-ago__text"}).text.strip()
            except: pass
            try: num_applicants_raw = job_soup.find("span", {"class": "num-applicants__caption"}).text.strip()
            except: pass
            try:
                insights = job_soup.find_all("li", {"class": "job-details-jobs-unified-top-card__job-insight"})
                for insight in insights:
                    if "💰" in insight.get_text():
                        salary_raw = insight.find("span").text.strip()
                        break
            except: pass
            
            job_post = {
                'job_link': f"https://www.linkedin.com/jobs/view/{job_id}",
                'job_title': job_soup.find("h2", {"class":"top-card-layout__title"}).text.strip() if job_soup.find("h2", {"class":"top-card-layout__title"}) else "N/A",
                'company_name': job_soup.find("a", {"class": "topcard__org-name-link"}).text.strip() if job_soup.find("a", {"class": "topcard__org-name-link"}) else "N/A",
                'salary': salary_raw,
                'job_desc': job_soup.find("div", {"class": "show-more-less-html__markup"}).get_text(separator="\n").strip() if job_soup.find("div", {"class": "show-more-less-html__markup"}) else "",
                'hours_posted': parse_time_posted(time_posted_raw),
                'applicants_count': parse_num_applicants(num_applicants_raw)
            }
            job_list.append(job_post)
        time.sleep(random.uniform(0.2, 0.8))
    return pd.DataFrame(job_list) if job_list else None

def process_nlp_features(df: pd.DataFrame, resume_text: str | None = None):
    if df is None or df.empty: return df
    df['top_keywords'] = df['job_desc'].apply(extract_keywords)
    df['extracted_skills'] = df['job_desc'].apply(lambda x: extract_skills(x, nlp_model))
    df['sentiment_score'] = df['job_desc'].apply(analyze_sentiment)
    if resume_text:
        df['resume_match'] = df['job_desc'].apply(lambda x: f"{calculate_match(x, resume_text) * 100:.2f}%")
    return df

# --- API Endpoint ---
@app.post("/analyze-jobs/", dependencies=[Depends(get_api_key)])
async def analyze_jobs_endpoint(request: JobRequest):
    """
    Scrapes LinkedIn jobs based on a title and location,
    performs NLP analysis, and returns the results.
    """
    print(f"Received request for title: {request.title}, location: {request.location}")
    
    # Run the scraper
    raw_df = run_linkedin_scraper(request.title, request.location, request.pages_to_scrape)
    if raw_df is None or raw_df.empty:
        raise HTTPException(status_code=404, detail="No jobs found for the given criteria.")
    
    # Run the NLP processing
    enhanced_df = process_nlp_features(raw_df, request.resume_text)
    
    # Convert DataFrame to a list of dictionaries (JSON-compatible)
    results = enhanced_df.to_dict(orient='records')
    
    return {"job_count": len(results), "jobs": results}