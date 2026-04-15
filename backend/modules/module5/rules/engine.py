from typing import List, Dict, Any
from .indication_rules import is_indicated
from .dose_rules import is_antibiotic, is_likely_viral


def run_rule_engine(diagnosis: str, drugs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    diagnosis = (diagnosis or "").lower().strip()

    results: List[Dict[str, Any]] = []

    for drug in drugs:
        name = (drug.get("generic") or drug.get("name") or "").lower().strip()

        rule_result = evaluate_drug_against_rules(diagnosis, name, drug)
        results.append(rule_result)

    return results


def evaluate_drug_against_rules(
    diagnosis: str,
    drug_name: str,
    drug_data: Dict[str, Any]
) -> Dict[str, Any]:

    # 1. Indication match
    if drug_name and is_indicated(diagnosis, drug_name):
        return {
            "drug": drug_data.get("name"),
            "status": "appropriate",
            "reason": f"{drug_name} is commonly indicated for {diagnosis}",
            "confidence": 0.9,
            "source": "rule:indication_match",
        }

    # 2. Antibiotic misuse
    if drug_name and is_antibiotic(drug_name) and is_likely_viral(diagnosis):
        return {
            "drug": drug_data.get("name"),
            "status": "inappropriate",
            "reason": "Antibiotic likely unnecessary for viral condition",
            "confidence": 0.95,
            "source": "rule:antibiotic_misuse",
        }

    # 3. Unknown → defer
    return {
        "drug": drug_data.get("name"),
        "status": "caution",
        "reason": "No strong rule match found",
        "confidence": 0.4,
        "source": "rule:unknown",
    }