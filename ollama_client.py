import os
import time
import requests

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "120"))


def check_ollama_ready(model: str = OLLAMA_MODEL) -> None:
    """Fail early with an actionable error if Ollama is down or missing the model."""
    try:
        resp = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Cannot reach Ollama at {OLLAMA_HOST}. "
            "Start the Ollama service or check OLLAMA_HOST (see README)."
        ) from exc

    installed = [m.get("name", "") for m in resp.json().get("models", [])]
    # Match either exact tag or base name (e.g., "llama3.2" matches "llama3.2:latest")
    if not any(name == model or name.startswith(f"{model}:") for name in installed):
        raise RuntimeError(
            f"Model '{model}' is not installed in Ollama. "
            f"Run: ollama pull {model}"
        )


def query_ollama(
    prompt: str,
    system: str | None = None,
    model: str = OLLAMA_MODEL,
    retries: int = 2,
) -> str:
    """Send a single inference request with basic retry handling."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }
    if system:
        payload["system"] = system

    last_err = None
    for attempt in range(retries + 1):
        try:
            resp = requests.post(
                f"{OLLAMA_HOST}/api/generate",
                json=payload,
                timeout=OLLAMA_TIMEOUT,
            )
            resp.raise_for_status()
            return resp.json().get("response", "").strip()
        except requests.RequestException as exc:
            last_err = exc
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))

    raise RuntimeError(f"Ollama request failed after {retries + 1} attempts") from last_err