import os
from serpapi import GoogleSearch
from dotenv import load_dotenv

load_dotenv()

SERP_KEY = os.getenv("SERP_API_KEY")

def search_web(query):

    params = {
        "q": query,
        "api_key": SERP_KEY,
        "engine": "google",
        "num": 5
    }

    search = GoogleSearch(params)
    results = search.get_dict()

    snippets = []

    if "organic_results" in results:
        for r in results["organic_results"][:5]:
            if "snippet" in r:
                snippets.append(r["snippet"])

    return "\n".join(snippets)