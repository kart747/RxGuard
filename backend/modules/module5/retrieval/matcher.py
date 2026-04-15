# backend/modules/module5/retrieval/matcher.py

from typing import Dict, Any, List
from .guideline_db import GUIDELINE_DB


def normalize_diagnosis(input_diag: str) -> str:
    input_diag = (input_diag or "").lower().strip()

    # 🔥 High-confidence keyword mapping (PRIMARY)
    if "pneumonia" in input_diag:
        return "pneumonia"

    if "uti" in input_diag or "urinary" in input_diag:
        return "urinary tract infection"

    if "viral" in input_diag or "flu" in input_diag or "cold" in input_diag:
        return "viral fever"

    # 🔁 fallback (simple overlap)
    input_words = set(input_diag.split())

    for canonical, data in GUIDELINE_DB.items():
        aliases = data.get("aliases", [])
        for alias in aliases:
            alias_words = set(alias.lower().split())
            if input_words.intersection(alias_words):
                return canonical

    return input_diag


def retrieve_guidelines(diagnosis: str, drugs: List[Dict[str, Any]]) -> Dict[str, Any]:
    normalized_diag = normalize_diagnosis(diagnosis)
    guideline = GUIDELINE_DB.get(normalized_diag)

    return {
        "normalized_diagnosis": normalized_diag,
        "guideline": guideline if guideline else {},
        "has_guideline": bool(guideline),
    }