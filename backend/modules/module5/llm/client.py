# backend/modules/module5/llm/client.py

from typing import List, Dict, Any
import json
import subprocess

from .prompt_builder import build_prompt


OLLAMA_MODEL = "biomistral:7b"


def call_ollama(prompt: str) -> str:
    """
    Calls Ollama locally via subprocess.
    Keeps it lightweight and dependency-free.
    """
    try:
        result = subprocess.run(
            ["ollama", "run", OLLAMA_MODEL],
            input=prompt.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
        )
        return result.stdout.decode("utf-8").strip()
    except Exception:
        return ""


def safe_json_parse(text: str) -> List[Dict[str, Any]]:
    """
    Extract and parse JSON safely.
    Prevents crashes from malformed LLM output.
    """
    try:
        # Attempt direct parse
        return json.loads(text)
    except Exception:
        # Try to extract JSON block
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1:
            try:
                return json.loads(text[start:end + 1])
            except Exception:
                return []
    return []


def run_llm_analysis(
    diagnosis: str,
    drugs: List[Dict[str, Any]],
    context: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    End-to-end LLM pipeline:
    prompt → ollama → parse → validate
    """

    prompt = build_prompt(diagnosis, drugs, context)

    raw_output = call_ollama(prompt)

    parsed = safe_json_parse(raw_output)

    # Minimal validation
    validated: List[Dict[str, Any]] = []

    for item in parsed:
        if not isinstance(item, dict):
            continue

        validated.append({
            "drug": item.get("drug"),
            "status": item.get("status", "caution"),
            "reason": item.get("reason", "LLM analysis"),
            "confidence": float(item.get("confidence", 0.6)),
            "source": "llm",
        })

    return validated