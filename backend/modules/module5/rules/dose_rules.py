# backend/modules/module5/rules/dose_rules.py

from typing import Set, List

ANTIBIOTICS: Set[str] = {
    "amoxicillin",
    "azithromycin",
    "ceftriaxone",
    "ciprofloxacin",
    "doxycycline",
}


def is_antibiotic(drug_name: str) -> bool:
    drug_name = (drug_name or "").lower().strip()
    return drug_name in ANTIBIOTICS


VIRAL_KEYWORDS: List[str] = [
    "viral",
    "flu",
    "cold",
    "dengue",
    "chikungunya",
]


def is_likely_viral(diagnosis: str) -> bool:
    diagnosis = (diagnosis or "").lower()
    return any(k in diagnosis for k in VIRAL_KEYWORDS)