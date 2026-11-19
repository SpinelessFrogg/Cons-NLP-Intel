import time
from datetime import datetime
import pandas as pd
import requests


def fetch_environmental_bills(query, limit, API_KEY, BASE_URL):
    # fetch legiscan metadata & action history
    # builds and gets request using params
    search = requests.get(BASE_URL, params={
        "key": API_KEY,
        "op": "getSearchRaw",
        "query": query
    }).json()

    if search.get("status") != "OK":
        raise RuntimeError(f"LegiScan API error: {search.get('alert', search)}")

    results = search.get("searchresult", {})
    raw_items = results.get("results", [])
    bill_ids = [item["bill_id"] for item in raw_items][:limit]

    records = []
    # get individual bill record data
    for id in bill_ids:
        bill_data = requests.get(BASE_URL, params={
            "key": API_KEY,
            "op": "getBill",
            "id": id
        }).json().get("bill", {})

        if not bill_data:
            continue

        history = bill_data.get("history", [])
        if history:
          # strips dates from action history to find active time and amount of actions
            dates = [datetime.strptime(h["date"], "%Y-%m-%d") for h in history if "date" in h]
            days_active = (max(dates) - min(dates)).days if len(dates) > 1 else 0
            action_count = len(history)
        else:
            days_active, action_count = 0, 0

        records.append({
            "bill_id": id, # legiscan PK
            "state": bill_data.get("state"), # 2 letter abbr. for US state, US if federal
            "bill_number": bill_data.get("bill_number"), # official legislative code, HB#### for house and SB#### for senate
            "status": bill_data.get("status"),  # 1=intro, 2=engrossed, 3=enrolled, 4=passed, 5=vetoed, 6=dead
            "title": bill_data.get("title", ""), # official string title of bill
            "description": bill_data.get("description", ""), # official legislative summary of the bill
            "days_active": days_active, # time between first introduction and last action
            "action_count": action_count, # number of actions taken on bill
            "url": bill_data.get("url") # link to bill page
        })
        time.sleep(0.1)  # rate limiting

    return pd.DataFrame(records)
