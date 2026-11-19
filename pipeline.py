import os

from api_call import fetch_environmental_bills
from nlpclustering import cluster_similar_bills
from extract_metadata import extract_policy_metadata
from graph_dashboard import graph_dashboard
from ollama_client import check_ollama_ready

API_KEY = os.getenv("LEGISCAN_API_KEY")
if not API_KEY:
    raise ValueError("Please set the LEGISCAN_API_KEY environment variable.")
BASE_URL = "https://api.legiscan.com/"

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")

limit = 300
query = "conservation OR wildlife OR habitat"
datafile = "cons_legislative_intelligence.csv"
distance_threshold = 0.45

if __name__ == "__main__":
    print("Fetching environmental legislation from LegiScan...")
    bills_df = fetch_environmental_bills(query=query, limit=limit, API_KEY=API_KEY, BASE_URL=BASE_URL)

    print("Clustering cross-state model bills via SentenceTransformers...")
    bills_df = cluster_similar_bills(bills_df, distance_threshold)

    print(f"Checking Ollama ({OLLAMA_MODEL} @ {OLLAMA_HOST}) and extracting structured policy intent...")
    check_ollama_ready(model=OLLAMA_MODEL)
    final_df = extract_policy_metadata(bills_df)    

    final_df.to_csv(datafile, index=False)
    print(f"Saved {len(final_df)} bills across {final_df['policy_cluster'].nunique()} clusters.")

    print("Graphing with Matplotlib...")
    graph_dashboard(final_df, datafile)