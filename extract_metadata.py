
from ollama import Client
from tqdm.auto import tqdm
import pandas as pd
from policy_classification import PolicyClassification
import os

DEFAULT_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
DEFAULT_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "60.0"))

def check_ollama_ready(client, model = DEFAULT_MODEL):
    # fail fast before looping if ollama isnt reachable
    try:
        listed = client.list()
    except Exception as exc:
        raise RuntimeError(
            f"Cannot reach Ollama at {client._client.base_url}. "
            "Start Ollama or set OLLAMA_HOST (see README)."
        ) from exc

    raw_models = getattr(listed, "models", None)
    if raw_models is None and isinstance(listed, dict):
        raw_models = listed.get("models", [])

    installed = []
    for m in raw_models or []:
        name = getattr(m, "model", None) or getattr(m, "name", None) or (m.get("model") or m.get("name") if isinstance(m, dict) else "")
        if name:
            installed.append(name)

    if not any(name == model or name.startswith(f"{model}:") for name in installed):
        raise RuntimeError(
            f"Ollama model '{model}' is not installed. Run: ollama pull {model}"
        )
    
def extract_policy_metadata(df, host=DEFAULT_HOST, model=DEFAULT_MODEL, timeout=DEFAULT_TIMEOUT, verify_ready=True):
    # extract regulatory fields with ollama & pydantic
    client = Client(host=host, timeout=timeout)
    if verify_ready:
        check_ollama_ready(client, model=model)

    extracted_rows = []
    schema = PolicyClassification.model_json_schema()
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Classifying bills"):
        desc = row["description"] if pd.notna(row["description"]) and row["description"].strip() else row["title"]
        prompt = (
            f"Classify this U.S. conservation bill and return ONLY valid JSON matching the schema. "
            f"Choose the most specific policy_mechanism possible; only use 'Other' if none fit.\n"
            f"State: {row['state']} | Bill: {row['bill_number']}\n"
            f"Title: {row['title']}\n"
            f"Description: {desc[:800]}"
        )
        try:
            response = client.chat(
                model="llama3.1",
                messages=[{"role": "user", "content": prompt}],
                format=schema,
                options={
                    "temperature": 0.1,
                    "repeat_penalty": 1.15,  # SAFEGUARD 4: Breaks whitespace/newline loops
                    "num_ctx": 2048,         #
                    "num_predict": 120,      # SAFEGUARD 5: Caps output at 120 tokens (~1 sec)
                },
            )
            parsed = PolicyClassification.model_validate_json(response.message.content)
            extracted_rows.append(parsed.model_dump())
        except Exception as e:
            print(f"\nRow {idx} ({row['bill_number']}) fallback due to: {e}")
            extracted_rows.append({
                "policy_mechanism": "Other",
                "target_sector": "Other",
                "advocacy_alignment": "Neutral / Procedural",
                "plain_english_summary": row["title"][:150],
            })

    meta_df = pd.DataFrame(extracted_rows)
    return pd.concat([df.reset_index(drop=True), meta_df], axis=1) # add new columns to master dataframe