# backend/modules/module5/rules/indication_rules.py

from typing import Dict, List

INDICATION_RULES: Dict[str, List[str]] = {
    "pneumonia": ["amoxicillin", "azithromycin", "ceftriaxone"],
    "urinary tract infection": ["nitrofurantoin", "ciprofloxacin"],
    "fever": [],  # intentionally vague
    "viral fever": [],
}


def is_indicated(diagnosis: str, drug_name: str) -> bool:
    diagnosis = (diagnosis or "").lower().strip()
    drug_name = (drug_name or "").lower().strip()

    drugs = INDICATION_RULES.get(diagnosis, [])
    return drug_name in drugs