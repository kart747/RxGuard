#!/usr/bin/env python3
"""
Module 5: Drug Appropriateness Checker using local Ollama (BioMistral default).

Inputs:
- normalized drugs from Module 2 output shape
- diagnosis from Module 4 output (or direct diagnosis string)

Checks whether the prescribed drugs are clinically appropriate for the diagnosis.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("BIO_MISTRAL_MODEL", "biomistral:7b-instruct-q4_K_M")
DEFAULT_TIMEOUT = float(os.getenv("DRUG_APPROPRIATENESS_TIMEOUT", "90"))


def _extract_json_object(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty LLM response")

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError("No JSON object found in LLM response")

    parsed = json.loads(m.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("LLM response JSON is not an object")
    return parsed


def _call_ollama_chat(prompt: str, model: str, host: str, timeout_sec: float) -> str:
    url = host.rstrip("/") + "/api/chat"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "Return only valid JSON. No markdown.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "stream": False,
        "options": {"temperature": 0.1},
    }

    req = Request(
        url,
        method="POST",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    try:
        with urlopen(req, timeout=timeout_sec) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return str(data.get("message", {}).get("content", ""))
    except HTTPError as exc:
        raise RuntimeError(f"Ollama HTTP error: {exc.code} {exc.reason}") from exc
    except URLError as exc:
        raise RuntimeError("Unable to reach Ollama. Is it running?") from exc


def _flatten_normalized_drugs(normalized_drugs: dict[str, Any]) -> list[dict[str, Any]]:
    drugs = normalized_drugs.get("drugs", []) if isinstance(normalized_drugs, dict) else []
    out: list[dict[str, Any]] = []

    for d in drugs:
        if not isinstance(d, dict):
            continue
        out.append(
            {
                "name": d.get("corrected_name") or d.get("original"),
                "generic": d.get("generic_name"),
                "dose": (d.get("structured") or {}).get("dose") if isinstance(d.get("structured"), dict) else None,
                "route": (d.get("structured") or {}).get("route") if isinstance(d.get("structured"), dict) else None,
                "frequency": (d.get("structured") or {}).get("frequency") if isinstance(d.get("structured"), dict) else None,
                "duration": (d.get("structured") or {}).get("duration") if isinstance(d.get("structured"), dict) else None,
            }
        )

    return out


def _build_prompt(diagnosis: str, drugs: list[dict[str, Any]]) -> str:
    return (
        "You are a clinical pharmacology assistant. "
        "Determine if each prescribed drug is appropriate for the given diagnosis. "
        "Return strict JSON with keys: overall_appropriate (boolean), confidence (0..1), "
        "summary (string), per_drug (array), cautions (array), alternatives (array). "
        "Each per_drug item must contain: drug, appropriate (boolean), reason (string), "
        "dose_route_frequency_duration_ok (boolean).\n\n"
        f"Diagnosis: {diagnosis}\n"
        f"Prescribed drugs (normalized): {json.dumps(drugs, ensure_ascii=False)}\n"
    )


def check_drug_appropriateness(
    *,
    normalized_drugs: dict[str, Any],
    diagnosis: str,
    model: str = DEFAULT_MODEL,
    host: str = DEFAULT_OLLAMA_HOST,
    timeout_sec: float = DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    if not diagnosis or not diagnosis.strip():
        return {
            "status": "error",
            "error": "Diagnosis is required",
            "model": model,
            "overall_appropriate": False,
            "confidence": 0.0,
            "summary": "Missing diagnosis",
            "per_drug": [],
            "cautions": ["Missing diagnosis"],
            "alternatives": [],
        }

    drugs = _flatten_normalized_drugs(normalized_drugs)
    if not drugs:
        return {
            "status": "error",
            "error": "No normalized drugs provided",
            "model": model,
            "overall_appropriate": False,
            "confidence": 0.0,
            "summary": "No drugs to evaluate",
            "per_drug": [],
            "cautions": ["No drugs to evaluate"],
            "alternatives": [],
        }

    prompt = _build_prompt(diagnosis=diagnosis, drugs=drugs)

    try:
        content = _call_ollama_chat(prompt=prompt, model=model, host=host, timeout_sec=timeout_sec)
        parsed = _extract_json_object(content)

        overall = bool(parsed.get("overall_appropriate", False))
        confidence = float(parsed.get("confidence", 0.0))
        confidence = max(0.0, min(1.0, confidence))

        per_drug = parsed.get("per_drug", [])
        if not isinstance(per_drug, list):
            per_drug = []

        cautions = parsed.get("cautions", [])
        if not isinstance(cautions, list):
            cautions = [str(cautions)]

        alternatives = parsed.get("alternatives", [])
        if not isinstance(alternatives, list):
            alternatives = [str(alternatives)]

        return {
            "status": "success",
            "model": model,
            "overall_appropriate": overall,
            "confidence": confidence,
            "summary": str(parsed.get("summary", "")),
            "per_drug": per_drug,
            "cautions": [str(x) for x in cautions],
            "alternatives": [str(x) for x in alternatives],
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "error",
            "error": str(exc),
            "model": model,
            "overall_appropriate": False,
            "confidence": 0.0,
            "summary": "Appropriateness check failed",
            "per_drug": [],
            "cautions": [
                "Verify Ollama is running",
                "Verify model exists in local Ollama registry",
            ],
            "alternatives": [],
        }


def check_from_module_outputs(
    *,
    module2_normalized_drugs: dict[str, Any],
    module4_validation: dict[str, Any],
    model: str = DEFAULT_MODEL,
    host: str = DEFAULT_OLLAMA_HOST,
    timeout_sec: float = DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    diagnosis = str(module4_validation.get("diagnosis") or "").strip()
    if not diagnosis:
        diagnosis = str(module4_validation.get("input_diagnosis") or "").strip()

    return check_drug_appropriateness(
        normalized_drugs=module2_normalized_drugs,
        diagnosis=diagnosis,
        model=model,
        host=host,
        timeout_sec=timeout_sec,
    )


def _main() -> int:
    parser = argparse.ArgumentParser(description="Module 5 Drug Appropriateness Checker")
    parser.add_argument("--normalized-drugs-json", required=True, help="JSON string of Module 2 normalized_drugs")
    parser.add_argument("--diagnosis", required=True, help="Diagnosis string from Module 4")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model tag")
    parser.add_argument("--host", default=DEFAULT_OLLAMA_HOST, help="Ollama host")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="Request timeout")
    args = parser.parse_args()

    try:
        normalized = json.loads(args.normalized_drugs_json)
    except Exception as exc:  # noqa: BLE001
        print(
            json.dumps(
                {
                    "status": "error",
                    "error": f"Invalid normalized drugs JSON: {exc}",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 1

    result = check_drug_appropriateness(
        normalized_drugs=normalized,
        diagnosis=args.diagnosis,
        model=args.model,
        host=args.host,
        timeout_sec=args.timeout,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
