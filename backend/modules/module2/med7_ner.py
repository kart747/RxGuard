#!/usr/bin/env python3
"""
Module 3: Regex + spaCy en_core_web_sm extractor for prescription text.

Extracts:
- patient.name
- patient.age
- patient.gender
- diagnosis
- entities (all NER entities)
- drug_entities (regex + spaCy PRODUCT-based candidates)

The module attempts to load en_core_web_sm and gracefully degrades if unavailable.
"""

from __future__ import annotations

import json
import re
from typing import Any

from backend.modules.module3.drug_normalization import normalize_drug_list

_SPACY_PIPELINE: Any | None = None
_SPACY_ERROR: str | None = None


def _load_spacy_pipeline(model_name: str | None = None) -> tuple[Any | None, str | None, str | None]:
    global _SPACY_PIPELINE, _SPACY_ERROR

    if _SPACY_PIPELINE is not None:
        return _SPACY_PIPELINE, model_name or "cached", None
    if _SPACY_ERROR is not None:
        return None, None, _SPACY_ERROR

    try:
        import spacy
    except Exception as exc:  # noqa: BLE001
        _SPACY_ERROR = f"spacy import failed: {exc}"
        return None, None, _SPACY_ERROR

    preferred = [model_name or "en_core_web_sm", "en_core_web_sm"]

    seen: set[str] = set()
    for name in preferred:
        if not name or name in seen:
            continue
        seen.add(name)
        try:
            nlp = spacy.load(name)
            _SPACY_PIPELINE = nlp
            return _SPACY_PIPELINE, name, None
        except Exception:
            continue

    _SPACY_ERROR = (
        "spaCy model not found. Install en_core_web_sm, for example: "
        "python -m spacy download en_core_web_sm"
    )
    return None, None, _SPACY_ERROR


def _extract_patient_name(text: str) -> str | None:
    patterns = [
        r"\b(?:mr|mrs|ms|miss|me|pt|patient)\.?\s*[:\-]?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})",
        r"\bname\s*[:\-]\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})",
    ]
    for p in patterns:
        m = re.search(p, text, flags=re.IGNORECASE)
        if m:
            candidate = re.sub(r"\s+", " ", m.group(1)).strip()
            if len(candidate) >= 3:
                return candidate
    return None


def _extract_age_gender(text: str) -> tuple[int | None, str | None]:
    m = re.search(r"\b(\d{1,3})\s*/\s*([mMfF])\b", text)
    if m:
        age = int(m.group(1))
        gender = "male" if m.group(2).lower() == "m" else "female"
        return age, gender

    age_match = re.search(r"\bage\s*[:\-]?\s*(\d{1,3})\b", text, flags=re.IGNORECASE)
    age = int(age_match.group(1)) if age_match else None

    if re.search(r"\bmale\b", text, flags=re.IGNORECASE):
        gender = "male"
    elif re.search(r"\bfemale\b", text, flags=re.IGNORECASE):
        gender = "female"
    else:
        gender = None

    return age, gender


def _extract_diagnosis(text: str) -> str | None:
    patterns = [
        r"\b(?:diagnosis|dx)\s*[:\-]\s*(.+?)(?:\bRx\b|\bAdv\b|$)",
        r"\b(?:c/o|complaint|complaints)\s*[:\-]?\s*(.+?)(?:\bRx\b|\bAdv\b|$)",
    ]
    for p in patterns:
        m = re.search(p, text, flags=re.IGNORECASE)
        if m:
            value = re.sub(r"\s+", " ", m.group(1)).strip(" .,")
            if value:
                return value
    return None


def _regex_drug_entities(text: str) -> list[dict[str, Any]]:
    entities: list[dict[str, Any]] = []
    skip_tokens = {
        "tab",
        "tablet",
        "cap",
        "capsule",
        "inj",
        "injection",
        "syp",
        "syrup",
        "cream",
        "drop",
        "ointment",
        "after",
        "before",
        "meal",
        "meals",
        "days",
        "day",
        "for",
        "adv",
        "rx",
        "bd",
        "bid",
        "tid",
        "qid",
        "od",
        "qd",
        "x",
    }

    normalized = normalize_drug_list(text, use_rxnorm=False)
    allowed_tokens = {
        str(d.get("original", "")).strip().lower()
        for d in normalized.get("drugs", [])
        if d.get("original")
    }
    allowed_tokens.update(
        {
            str(d.get("corrected_name", "")).strip().lower()
            for d in normalized.get("drugs", [])
            if d.get("corrected_name")
        }
    )

    # Pattern: Tab./Cap./Inj. <DrugName>
    pattern = re.compile(
        r"\b(?:tab(?:let)?|cap(?:sule)?|inj(?:ection)?|syp|syrup|cream|drop|ointment)\.?\s*[:\-]?\s*([A-Za-z][A-Za-z0-9+\-/]{2,})",
        flags=re.IGNORECASE,
    )
    for m in pattern.finditer(text):
        token = m.group(1).strip(" .,")
        if allowed_tokens and token.lower() not in allowed_tokens:
            continue
        entities.append(
            {
                "text": token,
                "label": "DRUG_REGEX",
                "start": int(m.start(1)),
                "end": int(m.end(1)),
            }
        )

    # Pattern: Rx block fallbacks for single-word medicine-like tokens.
    rx_match = re.search(r"\bRx\b[:,\-]?\s*(.+?)(?:\bAdv\b|$)", text, flags=re.IGNORECASE)
    if rx_match:
        segment = rx_match.group(1)
        base = rx_match.start(1)
        for m in re.finditer(r"\b([A-Za-z][A-Za-z0-9+\-/]{2,})\b", segment):
            token = m.group(1)
            low = token.lower()
            if low in skip_tokens:
                continue
            if re.fullmatch(r"\d+", token):
                continue
            if allowed_tokens and low not in allowed_tokens:
                continue
            entities.append(
                {
                    "text": token,
                    "label": "DRUG_RX_BLOCK",
                    "start": int(base + m.start(1)),
                    "end": int(base + m.end(1)),
                }
            )

    dedup: dict[tuple[str, int, int], dict[str, Any]] = {}
    for item in entities:
        key = (item["text"].lower(), item["start"], item["end"])
        dedup[key] = item
    return list(dedup.values())


def extract_med7_entities(corrected_text: str, model_name: str | None = None) -> dict[str, Any]:
    """
    Backward-compatible function name.
    Uses regex + spaCy en_core_web_sm to extract patient/diagnosis/drug entities.
    """
    text = corrected_text or ""

    patient_name = _extract_patient_name(text)
    age, gender = _extract_age_gender(text)
    diagnosis = _extract_diagnosis(text)

    nlp, loaded_model, err = _load_spacy_pipeline(model_name=model_name or "en_core_web_sm")
    if nlp is None:
        return {
            "status": "unavailable",
            "model": loaded_model,
            "error": err,
            "patient": {
                "name": patient_name,
                "age": age,
                "gender": gender,
            },
            "diagnosis": diagnosis,
            "entities": [],
            "drug_entities": _regex_drug_entities(text),
        }

    doc = nlp(text)

    entities: list[dict[str, Any]] = []
    drug_entities: list[dict[str, Any]] = _regex_drug_entities(text)

    for ent in doc.ents:
        item = {
            "text": ent.text,
            "label": ent.label_,
            "start": int(ent.start_char),
            "end": int(ent.end_char),
        }
        entities.append(item)
        if ent.label_.upper() == "PRODUCT":
            if any(k in ent.text.lower() for k in ("tab", "cap", "inj", "syp")):
                drug_entities.append(item)

    # If regex missed patient name, prefer spaCy PERSON entities.
    if patient_name is None:
        for ent in doc.ents:
            if ent.label_ == "PERSON" and len(ent.text.strip()) >= 3:
                patient_name = ent.text.strip()
                break

    dedup_drugs: dict[tuple[str, int, int], dict[str, Any]] = {}
    for item in drug_entities:
        key = (item["text"].strip().lower(), item["start"], item["end"])
        dedup_drugs[key] = item
    drug_entities = list(dedup_drugs.values())

    return {
        "status": "success",
        "model": loaded_model,
        "error": None,
        "patient": {
            "name": patient_name,
            "age": age,
            "gender": gender,
        },
        "diagnosis": diagnosis,
        "entities": entities,
        "drug_entities": drug_entities,
    }


def _main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Module 3 spaCy + regex extractor")
    parser.add_argument("--text", required=True, help="corrected_text from OCR output")
    parser.add_argument("--model", default="en_core_web_sm", help="spaCy model name")
    args = parser.parse_args()

    result = extract_med7_entities(args.text, model_name=args.model)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
