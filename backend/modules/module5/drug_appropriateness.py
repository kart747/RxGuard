#!/usr/bin/env python3
"""
Module 5: Drug Appropriateness (Orchestrator)
"""

from __future__ import annotations
from .rules.engine import run_rule_engine
from .retrieval.matcher import retrieve_guidelines
from .llm.client import run_llm_analysis
from .scoring.confidence import apply_confidence

from typing import Any, List, Dict
import json


# =========================================================
# 🔹 Input Normalization
# =========================================================

def normalize_input_drugs(normalized_drugs: dict[str, Any]) -> List[Dict[str, Any]]:
    drugs = normalized_drugs.get("drugs", []) if isinstance(normalized_drugs, dict) else []
    out: List[Dict[str, Any]] = []

    for d in drugs:
        if not isinstance(d, dict):
            continue

        structured = d.get("structured") if isinstance(d.get("structured"), dict) else {}

        out.append(
            {
                "name": d.get("corrected_name") or d.get("original"),
                "generic": d.get("generic_name"),
                "dose": structured.get("dose"),
                "route": structured.get("route"),
                "frequency": structured.get("frequency"),
                "duration": structured.get("duration"),
            }
        )

    return out


# =========================================================
# 🔹 LLM Decision Control
# =========================================================

def should_call_llm(rule_results: List[Dict[str, Any]]) -> bool:
    for r in rule_results:
        if r.get("status") in ("unknown", None):
            return True
        if r.get("confidence", 0) < 0.75:
            return True
    return False


# =========================================================
# 🔹 Result Combination
# =========================================================

def combine_results(rule_results, llm_results):
    if not llm_results:
        return rule_results

    final = []

    for i, rule in enumerate(rule_results):
        if i < len(llm_results):
            llm = llm_results[i]

            if rule.get("confidence", 0) >= 0.8:
                final.append(rule)
            else:
                final.append(llm)
        else:
            final.append(rule)

    return final


# =========================================================
# 🔹 Output Formatter
# =========================================================

def format_output(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "drug_evaluation": results
    }


# =========================================================
# 🔹 Error Helper
# =========================================================

def _error(message: str) -> Dict[str, Any]:
    return {
        "drug_evaluation": [],
        "error": message
    }


# =========================================================
# 🔹 Main Orchestrator
# =========================================================

def check_drug_appropriateness(
    *,
    normalized_drugs: dict[str, Any],
    diagnosis: str,
) -> Dict[str, Any]:

    if not diagnosis or not diagnosis.strip():
        return _error("Missing diagnosis")

    drugs = normalize_input_drugs(normalized_drugs)

    if not drugs:
        return _error("No drugs to evaluate")

    # 1. Rules
    rule_results = run_rule_engine(diagnosis, drugs)

    # 2. Retrieval
    guideline_context = retrieve_guidelines(diagnosis, drugs)

    # 3. LLM decision
    needs_llm = should_call_llm(rule_results)

    llm_results = None
    if needs_llm and guideline_context.get("has_guideline"):
        llm_results = run_llm_analysis(
            diagnosis=guideline_context.get("normalized_diagnosis", diagnosis),
            drugs=drugs,
            context=guideline_context
        )

    # 4. Combine
    combined = combine_results(rule_results, llm_results)

    # 5. Confidence normalization
    final_results = apply_confidence(combined)

    # 6. Output
    return format_output(final_results)


# =========================================================
# 🔹 Integration Helper
# =========================================================

def check_from_module_outputs(
    *,
    module2_normalized_drugs: dict[str, Any],
    module4_validation: dict[str, Any],
) -> Dict[str, Any]:

    diagnosis = str(module4_validation.get("diagnosis") or "").strip()

    if not diagnosis:
        diagnosis = str(module4_validation.get("input_diagnosis") or "").strip()

    return check_drug_appropriateness(
        normalized_drugs=module2_normalized_drugs,
        diagnosis=diagnosis,
    )


# =========================================================
# 🔹 CLI
# =========================================================

def _main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Module 5 Drug Appropriateness Checker")
    parser.add_argument("--normalized-drugs-json", required=True)
    parser.add_argument("--diagnosis", required=True)
    args = parser.parse_args()

    try:
        normalized = json.loads(args.normalized_drugs_json)
    except Exception as exc:
        print(json.dumps({"error": f"Invalid JSON: {exc}"}, indent=2))
        return 1

    result = check_drug_appropriateness(
        normalized_drugs=normalized,
        diagnosis=args.diagnosis,
    )

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())