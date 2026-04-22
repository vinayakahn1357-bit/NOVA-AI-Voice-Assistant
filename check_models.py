"""
Model health check — tests every configured model via the NOVA API layer.
Run from NOVAbackend root: python check_models.py
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(__file__))

import requests as req_lib
from requests.adapters import HTTPAdapter

os.environ.setdefault("NOVA_ENV", "production")

from config import get_settings, GROQ_API_URL, NVIDIA_API_URL

settings = get_settings()
GROQ_KEY   = settings.get("groq_api_key", "")
NVIDIA_KEY = settings.get("nvidia_api_key", "")

TEST_MSG = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user",   "content": "Say 'OK' and nothing else."},
]

http = req_lib.Session()
http.mount("https://", HTTPAdapter(pool_connections=4, pool_maxsize=8))  # type: ignore[arg-type]

TIMEOUT = 30

GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "llama3-70b-8192",
    "llama3-8b-8192",
    "gemma2-9b-it",
]

NVIDIA_MODELS = [
    "nvidia/llama-3.3-70b-instruct",
    "nvidia/llama-3.1-nemotron-70b-instruct",
    "meta/llama-3.1-405b-instruct",
    "meta/llama-3.1-70b-instruct",
    "mistralai/mistral-large-2-instruct",
    "mistralai/mixtral-8x22b-instruct-v0.1",
    "google/gemma-3-27b-it",
    "microsoft/phi-4",
]

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"
BOLD   = "\033[1m"


def test_model(provider, model, url, key):
    t0 = time.time()
    try:
        r = http.post(
            url,
            json={
                "model": model,
                "messages": TEST_MSG,
                "temperature": 0.1,
                "max_tokens": 10,
                "stream": False,
            },
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            timeout=TIMEOUT,
        )
        elapsed = round(time.time() - t0, 2)
        if r.status_code == 200:
            try:
                content = r.json()["choices"][0]["message"]["content"].strip()[:60]
            except Exception:
                content = "(parse error)"
            return True, elapsed, content
        else:
            try:
                err = r.json().get("error", {}).get("message", r.text[:120])
            except Exception:
                err = r.text[:120]
            return False, elapsed, f"HTTP {r.status_code}: {err}"
    except Exception as exc:
        elapsed = round(time.time() - t0, 2)
        return False, elapsed, str(exc)[:120]


# ── GROQ ──────────────────────────────────────────────────────────────────────
groq_results: dict[str, bool] = {}
nvidia_results: dict[str, bool] = {}
print(f"\n{BOLD}=== Groq Models ==={RESET}")
if not GROQ_KEY:
    print(f"  {YELLOW}SKIP — no Groq API key configured{RESET}")
else:
    groq_results = {}
    for model in GROQ_MODELS:
        ok, t, msg = test_model("groq", model, GROQ_API_URL, GROQ_KEY)
        status = f"{GREEN}✓ PASS{RESET}" if ok else f"{RED}✗ FAIL{RESET}"
        print(f"  {status}  {model:<45}  {t:>5}s  {msg}")
        groq_results[model] = ok

# ── NVIDIA ─────────────────────────────────────────────────────────────────────
print(f"\n{BOLD}=== NVIDIA NIM Models ==={RESET}")
if not NVIDIA_KEY:
    print(f"  {YELLOW}SKIP — no NVIDIA API key configured{RESET}")
else:
    nvidia_results = {}
    for model in NVIDIA_MODELS:
        ok, t, msg = test_model("nvidia", model, NVIDIA_API_URL, NVIDIA_KEY)
        status = f"{GREEN}✓ PASS{RESET}" if ok else f"{RED}✗ FAIL{RESET}"
        print(f"  {status}  {model:<50}  {t:>5}s  {msg}")
        nvidia_results[model] = ok

# ── Summary ────────────────────────────────────────────────────────────────────
print(f"\n{BOLD}=== Summary ==={RESET}")
if GROQ_KEY:
    g_pass = sum(groq_results.values())
    g_total = len(groq_results)
    print(f"  Groq:   {GREEN}{g_pass}/{g_total} passed{RESET}")
    if g_pass < g_total:
        failed = [m for m, ok in groq_results.items() if not ok]
        print(f"  {RED}Failed Groq:{RESET} {', '.join(failed)}")

if NVIDIA_KEY:
    n_pass = sum(nvidia_results.values())
    n_total = len(nvidia_results)
    print(f"  NVIDIA: {GREEN}{n_pass}/{n_total} passed{RESET}")
    if n_pass < n_total:
        failed = [m for m, ok in nvidia_results.items() if not ok]
        print(f"  {RED}Failed NVIDIA:{RESET} {', '.join(failed)}")
print()
