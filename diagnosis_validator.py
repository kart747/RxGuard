#!/usr/bin/env python3
"""
Module 4: Diagnosis Validator using local Ollama + BioMistral-7B Q4.

Purpose:
- Validate whether a diagnosis is clinically plausible from structured NER output.
- Use BioMistral model served locally through Ollama.

Input fields:
- patient symptoms
- age
- gender
- examination findings
- diagnosis

Output fields:
- is_plausible: bool
- confidence: float (0..1)
- rationale: str
- red_flags: list[str]
- suggested_checks: list[str]
- differential_diagnoses: list[str]
- status: success|error
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
DEFAULT_TIMEOUT = float(os.getenv("DIAG_VALIDATOR_TIMEOUT", "90"))


def _extract_json_object(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty LLM response")

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in LLM response")

    data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("LLM JSON response is not an object")
    return data


def _heuristic_red_flags(
    symptoms: str,
    age: int | None,
    gender: str | None,
    diagnosis: str,
) -> list[str]:
    flags: list[str] = []
    d = (diagnosis or "").lower()
    g = (gender or "").lower().strip()

    if "pregnan" in d and g == "male":
        flags.append("Diagnosis references pregnancy for male patient")
    if any(k in d for k in ("prostate", "bph", "prostatitis")) and g == "female":
        flags.append("Diagnosis references prostate-related disease for female patient")
    if any(k in d for k in ("menopause", "ovarian", "uterine")) and g == "male":
        flags.append("Diagnosis references gynecologic condition for male patient")

    if age is not None and age < 12 and "alzheimer" in d:
        flags.append("Alzheimer diagnosis is atypical in pediatric age")

    s = (symptoms or "").lower()
    if diagnosis and symptoms and len(s) > 0:
        if "fracture" in d and all(k not in s for k in ("trauma", "fall", "pain", "injury")):
            flags.append("Fracture diagnosis without supportive trauma-like symptoms")

    return flags


def _build_prompt(
    symptoms: str,
    age: int | None,
    gender: str | None,
    examination_findings: str,
    diagnosis: str,
    heuristic_flags: list[str],
) -> str:
    return (
        "You are a clinical diagnosis validation assistant. "
        "Given patient summary fields, decide whether the diagnosis is clinically plausible. "
        "Do NOT provide treatment. Respond ONLY as strict JSON with keys: "
        "is_plausible (boolean), confidence (0..1), rationale (string), "
        "red_flags (array of strings), suggested_checks (array of strings), "
        "differential_diagnoses (array of strings).\n\n"
        f"Symptoms: {symptoms or 'unknown'}\n"
        f"Age: {age if age is not None else 'unknown'}\n"
        f"Gender: {gender or 'unknown'}\n"
        f"Examination findings: {examination_findings or 'unknown'}\n"
        f"Diagnosis: {diagnosis or 'unknown'}\n"
        f"Pre-check heuristic red flags: {heuristic_flags if heuristic_flags else []}\n"
    )


def _call_ollama_chat(
    prompt: str,
    model: str,
    host: str,
    timeout_sec: float,
) -> str:
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
        "options": {
            "temperature": 0.1,
        },
    }

    req = Request(
        url,
        method="POST",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    try:
        with urlopen(req, timeout=timeout_sec) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
        msg = raw.get("message", {})
        content = msg.get("content", "")
        return str(content)
    except HTTPError as exc:
        raise RuntimeError(f"Ollama HTTP error: {exc.code} {exc.reason}") from exc
    except URLError as exc:
        raise RuntimeError(
            "Unable to reach Ollama. Ensure Ollama is running and host is correct."
        ) from exc


def validate_diagnosis(
    *,
    symptoms: str,
    age: int | None,
    gender: str | None,
    examination_findings: str,
    diagnosis: str,
    model: str = DEFAULT_MODEL,
    host: str = DEFAULT_OLLAMA_HOST,
    timeout_sec: float = DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    """Validate diagnosis plausibility using local Ollama + BioMistral."""
    if not diagnosis or not diagnosis.strip():
        return {
            "status": "error",
            "error": "Diagnosis is required",
            "is_plausible": False,
            "confidence": 0.0,
            "rationale": "Missing diagnosis",
            "red_flags": ["Missing diagnosis"],
            "suggested_checks": [],
            "differential_diagnoses": [],
            "model": model,
            "input_diagnosis": diagnosis,
        }

    heuristic_flags = _heuristic_red_flags(symptoms, age, gender, diagnosis)
    prompt = _build_prompt(
        symptoms=symptoms,
        age=age,
        gender=gender,
        examination_findings=examination_findings,
        diagnosis=diagnosis,
        heuristic_flags=heuristic_flags,
    )

    try:
        content = _call_ollama_chat(
            prompt=prompt,
            model=model,
            host=host,
            timeout_sec=timeout_sec,
        )
        data = _extract_json_object(content)

        is_plausible = bool(data.get("is_plausible", False))
        confidence = float(data.get("confidence", 0.0))
        confidence = max(0.0, min(1.0, confidence))

        red_flags = data.get("red_flags", [])
        if not isinstance(red_flags, list):
            red_flags = [str(red_flags)]
        red_flags = [str(x) for x in red_flags] + heuristic_flags

        suggested = data.get("suggested_checks", [])
        if not isinstance(suggested, list):
            suggested = [str(suggested)]

        differentials = data.get("differential_diagnoses", [])
        if not isinstance(differentials, list):
            differentials = [str(differentials)]

        return {
            "status": "success",
            "model": model,
            "input_diagnosis": diagnosis,
            "is_plausible": is_plausible,
            "confidence": confidence,
            "rationale": str(data.get("rationale", "")),
            "red_flags": sorted(set(red_flags)),
            "suggested_checks": [str(x) for x in suggested],
            "differential_diagnoses": [str(x) for x in differentials],
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "error",
            "model": model,
            "input_diagnosis": diagnosis,
            "error": str(exc),
            "is_plausible": False,
            "confidence": 0.0,
            "rationale": "Validation failed",
            "red_flags": heuristic_flags,
            "suggested_checks": [
                "Verify that Ollama is running",
                "Verify model is installed: ollama list",
                f"Ensure model tag exists: {model}",
            ],
            "differential_diagnoses": [],
        }


def validate_from_ner_output(
    ner_output: dict[str, Any],
    *,
    symptoms: str = "",
    examination_findings: str = "",
    model: str = DEFAULT_MODEL,
    host: str = DEFAULT_OLLAMA_HOST,
    timeout_sec: float = DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    """
    Helper that consumes Module 3 style NER output.

    Expected minimal shape:
    {
      "patient": {"age": int|None, "gender": str|None},
      "diagnosis": str|None,
      ...
    }
    """
    patient = ner_output.get("patient", {}) if isinstance(ner_output, dict) else {}
    age_val = patient.get("age")
    age: int | None
    try:
        age = int(age_val) if age_val is not None else None
    except Exception:
        age = None

    gender = patient.get("gender") if isinstance(patient, dict) else None
    diagnosis = ner_output.get("diagnosis") if isinstance(ner_output, dict) else None

    return validate_diagnosis(
        symptoms=symptoms,
        age=age,
        gender=str(gender) if gender is not None else None,
        examination_findings=examination_findings,
        diagnosis=str(diagnosis) if diagnosis is not None else "",
        model=model,
        host=host,
        timeout_sec=timeout_sec,
    )


def _main() -> int:
    parser = argparse.ArgumentParser(description="Module 4 Diagnosis Validator")
    parser.add_argument("--symptoms", default="", help="Patient symptoms text")
    parser.add_argument("--age", type=int, default=None, help="Patient age")
    parser.add_argument("--gender", default=None, help="Patient gender")
    parser.add_argument("--exam", default="", help="Examination findings")
    parser.add_argument("--diagnosis", required=True, help="Diagnosis to validate")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model tag")
    parser.add_argument("--host", default=DEFAULT_OLLAMA_HOST, help="Ollama host")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="Request timeout")
    args = parser.parse_args()

    result = validate_diagnosis(
        symptoms=args.symptoms,
        age=args.age,
        gender=args.gender,
        examination_findings=args.exam,
        diagnosis=args.diagnosis,
        model=args.model,
        host=args.host,
        timeout_sec=args.timeout,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
