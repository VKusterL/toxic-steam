#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
up_llm_backends.py  -  unified multi-LLM caller (ollama / anthropic / openai)

Used by the narrative panel (4-model generation) and by the cross-model judge. Each model has a
backend; call_model(name, system, user) resolves it automatically. Determinism: temperature=0
where applicable (Opus 4.8 REMOVED temperature/top_p -> do not send them). Local models (Qwen/Llama)
run via Ollama = $0. Paid models (Opus/GPT-4o) run via API.

NB: the Batch API (-50%) is not implemented here (per-request is enough for ~50 users; the
estimated cost is already ~$3). For larger scale, switch to classify_claude_batch (Anthropic) /
OpenAI Batches -- see _archive_review_scope/src/sprint4_llm_classify.py.
"""
from __future__ import annotations
import os, time, json, re

# registry: short name -> backend + model id
MODELS = {
    "opus":  {"backend": "anthropic", "model": "claude-opus-4-8", "paid": True},
    "gpt4o": {"backend": "openai",    "model": "gpt-4o",          "paid": True},
    "qwen":  {"backend": "ollama",    "model": "qwen2.5:7b",      "paid": False},
    "llama": {"backend": "ollama",    "model": "llama3.1:8b",     "paid": False},
}
OLLAMA_HOST = "http://localhost:11434"

def call_ollama(model, system, user, timeout=240):
    import requests
    payload = {"model": model, "stream": False, "format": "json", "options": {"temperature": 0},
               "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    r = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=timeout); r.raise_for_status()
    return r.json()["message"]["content"]

def call_anthropic(model, system, user, max_tokens=512, client=None):
    import anthropic
    client = client or anthropic.Anthropic()
    # Opus 4.8/4.7 REMOVED temperature/top_p -> determinism only via prompt
    msg = client.messages.create(model=model, max_tokens=max_tokens, system=system,
                                 messages=[{"role": "user", "content": user}])
    return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")

def call_openai(model, system, user, max_tokens=512, client=None):
    from openai import OpenAI
    client = client or OpenAI()
    r = client.chat.completions.create(model=model, max_tokens=max_tokens, temperature=0,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
    return r.choices[0].message.content

def available(name):
    """Is the backend reachable? (Ollama server up / API key set)"""
    b = MODELS[name]["backend"]
    if b == "ollama":
        try:
            import requests; requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5).raise_for_status(); return True
        except Exception: return False
    if b == "anthropic": return bool(os.environ.get("ANTHROPIC_API_KEY"))
    if b == "openai": return bool(os.environ.get("OPENAI_API_KEY"))
    return False

def call_model(name, system, user, max_tokens=512, retries=3):
    spec = MODELS[name]; b = spec["backend"]; raw = ""
    for att in range(retries):
        try:
            if b == "ollama":    return call_ollama(spec["model"], system, user)
            if b == "anthropic": return call_anthropic(spec["model"], system, user, max_tokens)
            if b == "openai":    return call_openai(spec["model"], system, user, max_tokens)
        except Exception as e:
            raw = f"ERR:{e}"; time.sleep(1.5 * (att + 1))
    return raw

def parse_json(text):
    if not text: return {}
    try: return json.loads(text)
    except Exception: pass
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try: return json.loads(m.group(0))
        except Exception: pass
    return {}
