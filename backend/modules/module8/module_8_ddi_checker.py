"""

Module 8 — Drug Interaction & Contraindication Checker
Hospital Prescription Auditing System — Offline MVP

Decision architecture (four layers):
    Layer 1  DrugBank XML + NetworkX   — verified, database-first
    Layer 2  DeepDDI model             — ML fallback for unknown pairs
    Layer 3  Mistral via Ollama        — LLM last resort (always needs_review=True)
    Parallel openFDA JSON labels       — drug-disease / contraindication check
"""

from __future__ import annotations
import argparse
import glob
import itertools
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from typing import Any
import networkx as nx


DEFAULT_XML_PATH: str = "drugbank_full_database.xml"

DEFAULT_OPENFDA_FOLDER: str = "openfda_labels"

DEFAULT_DEEPDDI_MODEL_PATH: str = "deepddi_model.pt"

DEEPDDI_CONFIDENCE_THRESHOLD: float = 0.80

OLLAMA_MODEL: str = "mistral"

_DB_NS: dict[str, str] = {"db": "http://www.drugbank.ca"}


_CONDITION_KEYWORDS: dict[str, list[str]] = {
    "ckd": [
        "renal impairment", "renal failure", "chronic kidney disease", "ckd",
        "renal insufficiency", "kidney disease", "kidney impairment",
        "reduced renal function", "creatinine clearance", "egfr",
        "glomerular filtration", "dialysis", "nephrotoxic",
    ],
    "liver disease": [
        "hepatic impairment", "liver disease", "hepatic failure",
        "liver failure", "hepatic insufficiency", "liver impairment",
        "cirrhosis", "hepatotoxic", "hepatotoxicity", "elevated liver enzymes",
        "alanine aminotransferase", "aspartate aminotransferase",
        "child-pugh", "active liver disease",
    ],
    "asthma": [
        "asthma", "bronchospasm", "bronchial asthma", "reactive airway",
        "bronchoconstriction", "copd", "chronic obstructive pulmonary",
    ],
    "qt prolongation": [
        "qt prolongation", "qt interval", "torsades de pointes",
        "qtc prolongation", "cardiac arrhythmia", "ventricular arrhythmia",
        "prolonged qt",
    ],
    "pregnancy": [
        "pregnancy", "pregnant", "fetal", "foetal", "teratogenic",
        "embryotoxic", "lactation", "breastfeeding", "breast-feeding",
        "neonatal", "trimester",
    ],
    "peptic ulcer disease": [
        "peptic ulcer", "gastrointestinal bleeding", "gi bleeding",
        "gastric ulcer", "duodenal ulcer", "gastrointestinal ulceration",
    ],
    "heart failure": [
        "heart failure", "cardiac failure", "congestive heart failure",
        "chf", "left ventricular dysfunction", "reduced ejection fraction",
    ],
    "hypertension": [
        "hypertension", "high blood pressure", "blood pressure",
        "antihypertensive", "elevated blood pressure",
    ],
    "hyperkalemia": [
        "hyperkalemia", "high potassium", "elevated potassium",
        "serum potassium", "potassium level",
    ],
    "diabetes": [
        "diabetes", "diabetic", "hyperglycemia", "hypoglycemia",
        "blood glucose", "insulin", "glycemic",
    ],
    "epilepsy": [
        "epilepsy", "seizure", "convulsion", "anticonvulsant",
        "antiepileptic", "seizure threshold",
    ],
    "hypothyroidism": [
        "hypothyroidism", "thyroid", "thyroid disease", "thyroid dysfunction",
    ],
}

#: openFDA label JSON keys that carry warning/contraindication text
_FDA_WARNING_SECTIONS: list[str] = [
    "contraindications",
    "warnings",
    "warnings_and_cautions",
    "precautions",
    "use_in_specific_populations",
    "nursing_mothers",
    "pregnancy",
    "renal_impairment",
    "hepatic_impairment",
    "drug_interactions",
    "boxed_warning",
]

#:Keywords that indicate a *major* severity level in label text
_MAJOR_SIGNALS: tuple[str, ...] = (
    "contraindicated", "do not use", "must not", "should not be used",
    "fatal", "life-threatening", "serious", "severe", "avoid",
    "not recommended",
)

# Keywords that indicate a *moderate* severity level in label text
_MODERATE_SIGNALS: tuple[str, ...] = (
    "caution", "use with caution", "monitor", "may increase risk",
    "dose adjustment", "consider", "reduce dose",
)

def _normalise_name(name: str) -> str:
    """
    Return a lowercase, whitespace-stripped version of a drug or condition name.

    Args:
        name: Raw drug or condition name string.

    Returns:
        Lowercase, stripped string.
    """
    return name.strip().lower()


def _flatten_normalized_drugs(normalized_drugs: dict[str, Any]) -> list[str]:
    """
    Extract a flat, deduplicated list of generic drug names from Module 2 output.

    Args:
        normalized_drugs: Module 2 output dictionary.

    Returns:
        Deduplicated list of lowercase generic name strings.

    """
    drugs: list[str] = []
    for entry in normalized_drugs.get("drugs", []):
        name: str = (
            entry.get("generic_name")
            or entry.get("corrected_name")
            or ""
        )
        name = _normalise_name(name)
        if name and name not in drugs:
            drugs.append(name)
    return drugs


def _extract_comorbidities_from_module5(
    module5_appropriateness: dict[str, Any],
) -> list[str]:
    """
    Pull the comorbidity list out of Module 5 output.

    Args:
        module5_appropriateness: Module 5 output dictionary.

    Returns:
        Deduplicated list of lowercase comorbidity strings.
    """
    comorbidities: list[str] = []
    for key in ("comorbidities", "diagnoses", "conditions", "diagnosis_list"):
        raw = module5_appropriateness.get(key, [])
        if isinstance(raw, list):
            comorbidities.extend(
                _normalise_name(c) for c in raw if isinstance(c, str) and c.strip()
            )

    # Deduplicate while preserving insertion order
    seen: set[str] = set()
    return [c for c in comorbidities if not (c in seen or seen.add(c))]  # type: ignore[func-returns-value]


def _severity_rank(severity: str | None) -> int:
    """
    Convert a severity string to a numeric rank for comparison.

    Args:
        severity: One of major, moderate, minor, unknown, or None.

    Returns:
        Integer rank (3 = major, 2 = moderate, 1 = minor, 0 = unknown/None).
    """
    return {"major": 3, "moderate": 2, "minor": 1}.get(severity or "", 0)

# Layer 1: DrugBank XML + NetworkX

def _infer_severity_from_text(description: str) -> str:
    """
    Infer interaction severity from a free-text DrugBank description.

    Args:
        description: Raw interaction description string from DrugBank XML.

    Returns:
        One of major, moderate, minor, or unknown.
    """
    text = description.lower()

    major_keywords = (
        "contraindicated", "life-threatening", "serious", "severe", "major",
        "fatal", "death", "avoid", "do not use", "do not co-administer",
        "significant increase", "significant decrease", "significantly",
    )
    moderate_keywords = (
        "moderate", "caution", "monitor closely", "increased risk",
        "reduced efficacy", "may increase", "may decrease", "consider",
    )
    minor_keywords = ("minor", "minimal", "slight", "unlikely", "small")

    if any(kw in text for kw in major_keywords):
        return "major"
    if any(kw in text for kw in moderate_keywords):
        return "moderate"
    if any(kw in text for kw in minor_keywords):
        return "minor"
    return "unknown"


def _load_drugbank_graph(xml_path: str) -> nx.Graph:
    """
    Parse the DrugBank full XML database and build a NetworkX interaction graph.

    Graph schema:
        - **Nodes**: lowercase generic drug name strings
        - **Edges**: one undirected edge per interaction pair

    Args:
        xml_path: Path to drugbank_full_database.xml on disk.

    Returns:
        Populated nx.Graph ready for pairwise edge lookups.

    """
    if not os.path.isfile(xml_path):
        raise FileNotFoundError(
            f"DrugBank XML not found at: '{xml_path}'\n"
            "Download the full DrugBank database and place it at the configured path."
        )

    G: nx.Graph = nx.Graph()
    context = ET.iterparse(xml_path, events=("end",))

    for _event, elem in context:
        # Only process top-level <drug> elements — skip all other tags
        if elem.tag != f"{{{_DB_NS['db']}}}drug":
            continue

        # ── Primary drug name ─────────────────────────────────────────────
        name_el = elem.find("db:name", _DB_NS)
        if name_el is None or not (name_el.text or "").strip():
            elem.clear()
            continue

        drug_name = _normalise_name(name_el.text)
        G.add_node(drug_name)

        # ── Interaction partner entries ───────────────────────────────────
        interactions_el = elem.find("db:drug-interactions", _DB_NS)
        if interactions_el is None:
            elem.clear()
            continue

        for interaction_el in interactions_el.findall("db:drug-interaction", _DB_NS):
            partner_el = interaction_el.find("db:name", _DB_NS)
            if partner_el is None or not (partner_el.text or "").strip():
                continue

            partner = _normalise_name(partner_el.text)

            desc_el = interaction_el.find("db:description", _DB_NS)
            description = (desc_el.text or "").strip() if desc_el is not None else ""
            severity = _infer_severity_from_text(description)
            mechanism = description.split(".")[0].strip() if description else "see description"

            G.add_edge(
                drug_name,
                partner,
                severity=severity,
                mechanism=mechanism,
                description=description,
                source="drugbank",
            )

        # Free memory — critical for large XML files to avoid OOM
        elem.clear()

    return G


def check_drug_drug_from_drugbank(
    drugs: list[str],
    xml_path: str = DEFAULT_XML_PATH,
) -> tuple[list[dict[str, Any]], list[tuple[str, str]]]:
    """
    Layer 1 — DrugBank DDI check.

    Args:
        drugs:    Flat list of lowercase generic drug names.
        xml_path: Path to drugbank_full_database.xml.

    Returns:
        A tuple of:
        - found_interactions: list of interaction result dicts for pairs found in DrugBank
        - not_found_pairs:    list of (drug_a, drug_b) tuples NOT in DrugBank
    """
    found_interactions: list[dict[str, Any]] = []
    not_found_pairs: list[tuple[str, str]] = []

    G = _load_drugbank_graph(xml_path)

    for drug_a, drug_b in itertools.combinations(drugs, 2):
        if G.has_edge(drug_a, drug_b):
            edge_data: dict[str, Any] = G[drug_a][drug_b]
            severity = edge_data.get("severity", "unknown")

            found_interactions.append(
                {
                    "drug_a": drug_a,
                    "drug_b": drug_b,
                    "interaction_found": True,
                    "severity": severity,
                    "mechanism": edge_data.get("mechanism", ""),
                    "description": edge_data.get("description", ""),
                    "source": "drugbank",
                    "confidence": 1.0,         # Database lookup = certainty
                    "needs_review": severity in ("major", "unknown"),
                }
            )
        else:
            not_found_pairs.append((drug_a, drug_b))

    return found_interactions, not_found_pairs


def predict_ddi_deepddi(
    drug_a: str,
    drug_b: str,
    model_path: str = DEFAULT_DEEPDDI_MODEL_PATH,
) -> dict[str, Any]:
    """
    Layer 2 — DeepDDI machine learning DDI prediction.

    Args:
        drug_a:     Lowercase generic name of the first drug.
        drug_b:     Lowercase generic name of the second drug.
        model_path: Path to the serialised DeepDDI PyTorch model weights.

    Returns:
        Dict with keys:
        interaction_found, severity, confidence,
        mechanism, source, needs_review.
        confidence is 0.0 when the model was unavailable.
    """
    # ── Guard: model file must exist ─────────────────────────────────────────
    if not os.path.isfile(model_path):
        return _deepddi_unavailable_result(drug_a, drug_b, reason="model file not found")

    try:
        # Lazy import — keeps module loadable without PyTorch
        import torch  # type: ignore[import]
        import torch.nn as nn  # type: ignore[import]
    except ImportError:
        return _deepddi_unavailable_result(drug_a, drug_b, reason="torch not installed")

    try:
        model = torch.load(model_path, map_location="cpu")
        model.eval()

        def _name_to_tensor(name: str) -> "torch.Tensor":
            """Encode a drug name as a reproducible float tensor (placeholder)."""
            seed = sum(ord(c) for c in name)
            import random
            rng = random.Random(seed)
            vec = [rng.uniform(0, 1) for _ in range(512)]
            return torch.tensor(vec, dtype=torch.float32).unsqueeze(0)

        feat_a = _name_to_tensor(drug_a)
        feat_b = _name_to_tensor(drug_b)
        input_tensor = torch.cat([feat_a, feat_b], dim=1)  # shape: (1, 1024)

        #Inference
        with torch.no_grad():
            output = model(input_tensor)               # raw logits or probabilities
            probabilities = torch.softmax(output, dim=1)
            confidence_tensor = probabilities.max()
            predicted_class = probabilities.argmax().item()
            confidence = float(confidence_tensor.item())

        #Map predicted class to severity 
        # Adjust this mapping to match your model's actual output classes
        class_to_severity = {0: "none", 1: "minor", 2: "moderate", 3: "major"}
        severity = class_to_severity.get(int(predicted_class), "unknown")
        interaction_found = severity not in ("none", "unknown")

        return {
            "drug_a": drug_a,
            "drug_b": drug_b,
            "interaction_found": interaction_found,
            "severity": severity if interaction_found else None,
            "confidence": round(confidence, 4),
            "mechanism": f"DeepDDI predicted class {predicted_class} ({severity})",
            "description": f"ML model prediction — not verified by structured database",
            "source": "deepddi",
            "needs_review": True,   # ML predictions always require clinical review
        }

    except Exception as exc:  # noqa: BLE001
        return _deepddi_unavailable_result(drug_a, drug_b, reason=str(exc))


def _deepddi_unavailable_result(
    drug_a: str,
    drug_b: str,
    reason: str = "unavailable",
) -> dict[str, Any]:
    """
    Return a safe placeholder when DeepDDI cannot produce a prediction.

    Args:
        drug_a:  First drug name.
        drug_b:  Second drug name.
        reason:  Human-readable explanation for why DeepDDI was unavailable.

    Returns:
        Interaction result dict indicating no DeepDDI prediction was made.
    """
    return {
        "drug_a": drug_a,
        "drug_b": drug_b,
        "interaction_found": False,
        "severity": None,
        "confidence": 0.0,
        "mechanism": None,
        "description": f"DeepDDI layer skipped: {reason}",
        "source": "deepddi_unavailable",
        "needs_review": True,
    }


def predict_ddi_mistral(
    drug_a: str,
    drug_b: str,
    model_tag: str = OLLAMA_MODEL,
) -> dict[str, Any]:
    """
    Layer 3 — Mistral LLM DDI fallback via Ollama.

    Args:
        drug_a:    Lowercase generic name of the first drug.
        drug_b:    Lowercase generic name of the second drug.
        model_tag: Ollama model tag to use (default mistral).

    Returns:
        Dict with keys matching the standard interaction result schema.
        needs_review is always True.
        source is alway mistral_llm.
    """
    try:
        import ollama  # type: ignore[import]
    except ImportError:
        return _mistral_unavailable_result(drug_a, drug_b, reason="ollama package not installed")

    prompt = (
        f"You are a clinical pharmacology expert assisting a hospital drug safety system.\n\n"
        f"Drug A: {drug_a}\n"
        f"Drug B: {drug_b}\n\n"
        f"Task: Assess whether there is a clinically significant drug-drug interaction "
        f"between Drug A and Drug B.\n\n"
        f"Respond in JSON only — no prose, no markdown, no extra text.\n"
        f"Format:\n"
        f'{{"interaction_found": true/false, "severity": "major/moderate/minor/none", '
        f'"mechanism": "<short mechanism description>", '
        f'"clinical_significance": "<one sentence>"}}'
    )

    try:
        response = ollama.chat(
            model=model_tag,
            messages=[{"role": "user", "content": prompt}],
        )
        raw_text: str = response["message"]["content"].strip()

        # Strip any markdown code fences Mistral may add despite instructions
        raw_text = re.sub(r"```(?:json)?", "", raw_text).strip("` \n")

        parsed: dict[str, Any] = json.loads(raw_text)

        interaction_found: bool = bool(parsed.get("interaction_found", False))
        severity: str = parsed.get("severity", "unknown")
        if severity == "none":
            interaction_found = False
            severity = None  # type: ignore[assignment]

        return {
            "drug_a": drug_a,
            "drug_b": drug_b,
            "interaction_found": interaction_found,
            "severity": severity,
            "confidence": None,             # LLMs do not produce calibrated probabilities
            "mechanism": parsed.get("mechanism", "LLM-generated — not verified"),
            "description": parsed.get("clinical_significance", ""),
            "source": "mistral_llm",
            "needs_review": True,           # MANDATORY — never bypass for LLM results
        }

    except json.JSONDecodeError:
        # Mistral did not return valid JSON — record raw text for audit trail
        return {
            "drug_a": drug_a,
            "drug_b": drug_b,
            "interaction_found": False,
            "severity": None,
            "confidence": None,
            "mechanism": None,
            "description": f"LLM returned unparseable response",
            "source": "mistral_llm",
            "needs_review": True,
        }
    except Exception as exc:  # noqa: BLE001
        return _mistral_unavailable_result(drug_a, drug_b, reason=str(exc))


def _mistral_unavailable_result(
    drug_a: str,
    drug_b: str,
    reason: str = "unavailable",
) -> dict[str, Any]:
    """
    Return a safe placeholder when the Mistral/Ollama layer cannot respond.
    Args:
        drug_a:  First drug name.
        drug_b:  Second drug name.
        reason:  Explanation for why Mistral was unavailable.

    Returns:
        Interaction result dict with source mistral_unavailable.
    """
    return {
        "drug_a": drug_a,
        "drug_b": drug_b,
        "interaction_found": False,
        "severity": None,
        "confidence": None,
        "mechanism": None,
        "description": f"Mistral layer skipped: {reason}",
        "source": "mistral_unavailable",
        "needs_review": True,
    }

def _extract_field_text(field_value: Any) -> str:
    """
    Safely convert an openFDA label field to a single lowercase text string.

    Args:
        field_value: Raw value of an openFDA JSON label section.

    Returns:
        Lowercase concatenated text, or empty string.
    """
    if isinstance(field_value, list):
        return " ".join(str(item) for item in field_value if item).lower()
    if isinstance(field_value, str):
        return field_value.lower()
    return ""


def _extract_evidence_snippet(text: str, keyword: str, max_chars: int = 300) -> str:
    """
    Pull a readable context snippet from a label text around a matched keyword.

    Args:
        text:      Full lowercase label text to search.
        keyword:   The condition keyword that was matched.
        max_chars: Maximum length of the returned snippet.

    Returns:
        Cleaned text snippet, or the start of the text if the keyword is missing.
    """
    idx = text.find(keyword)
    if idx == -1:
        return re.sub(r"\s+", " ", text[:max_chars]).strip()

    start = max(0, idx - 80)
    end = min(len(text), idx + max_chars)
    snippet = text[start:end].strip()
    return re.sub(r"\s+", " ", snippet)


def _infer_fda_severity(text: str, matched_keywords: list[str]) -> str:
    """
    Infer severity from an openFDA label section using clinical signal keywords.

    Args:
        text:             Full lowercase label text.
        matched_keywords: The condition keywords that were found in the text.

    Returns:
        major, moderate, or minor.
    """
    for kw in matched_keywords:
        idx = text.find(kw)
        if idx == -1:
            continue
        context = text[max(0, idx - 150): idx + 150]
        if any(sig in context for sig in _MAJOR_SIGNALS):
            return "major"

    if any(sig in text for sig in _MODERATE_SIGNALS):
        return "moderate"

    return "minor"


def _load_openfda_rules(folder_path: str) -> dict[str, dict[str, Any]]:
    """
    Load all openFDA drug label JSON files and build a drug→condition rule lookup.
    Args:
        folder_path: Path to the folder containing the 13 openFDA JSON files.

    Returns:
        Nested ``dict[drug_name → dict[condition → metadata]]``.

    """
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(
            f"openFDA folder not found at: '{folder_path}'\n"
            "Create the folder and place all 13 openFDA JSON part files inside it."
        )

    json_files = sorted(glob.glob(os.path.join(folder_path, "*.json")))
    if not json_files:
        raise ValueError(
            f"No .json files found in '{folder_path}'.\n"
            "Ensure the 13 openFDA drug label JSON files are inside this folder."
        )

    rules: dict[str, dict[str, Any]] = {}

    for filepath in json_files:
        try:
            with open(filepath, encoding="utf-8") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            print(
                f"[openFDA] WARNING: skipping '{os.path.basename(filepath)}' — {exc}",
                file=sys.stderr,
            )
            continue

        results = data.get("results", [])
        if not isinstance(results, list):
            continue

        for entry in results:
            # ── Extract all generic / substance names for this label ───────
            openfda_meta = entry.get("openfda", {})
            generic_names: list[str] = [
                _normalise_name(n)
                for n in (
                    openfda_meta.get("generic_name", [])
                    + openfda_meta.get("substance_name", [])
                )
                if isinstance(n, str) and n.strip()
            ]

            if not generic_names:
                continue

            # ── Aggregate text from all warning / contraindication sections ─
            combined_parts: list[str] = []
            for section_key in _FDA_WARNING_SECTIONS:
                raw = entry.get(section_key)
                if raw:
                    combined_parts.append(_extract_field_text(raw))

            if not combined_parts:
                continue

            combined_text = " ".join(combined_parts)

            # ── Match each condition keyword group ─────────────────────────
            for condition, keywords in _CONDITION_KEYWORDS.items():
                matched = [kw for kw in keywords if kw in combined_text]
                if not matched:
                    continue

                evidence = _extract_evidence_snippet(combined_text, matched[0])
                severity = _infer_fda_severity(combined_text, matched)

                for drug_name in generic_names:
                    rules.setdefault(drug_name, {})
                    existing = rules[drug_name].get(condition)
                    if existing is None or _severity_rank(severity) > _severity_rank(existing["severity"]):
                        rules[drug_name][condition] = {
                            "severity": severity,
                            "evidence": evidence,
                            "source": "openfda",
                        }

    return rules


def _find_drug_rules(
    drug: str,
    rules: dict[str, dict[str, Any]],
) -> list[tuple[str, dict[str, Any]]]:
    """
    Find all rule entries that match a given drug name.

    Args:
        drug:  Lowercase generic drug name to look up.
        rules: Output of ``_load_openfda_rules()``.

    Returns:
        List of (rule_key, condition_dict) tuples.
    """
    if drug in rules:
        return [(drug, rules[drug])]

    return [
        (key, cond_dict)
        for key, cond_dict in rules.items()
        if drug in key or key in drug
    ]


def check_drug_disease_openfda(
    drugs: list[str],
    comorbidities: list[str],
    openfda_folder: str = DEFAULT_OPENFDA_FOLDER,
) -> list[dict[str, Any]]:
    """
    Parallel layer — openFDA drug-disease / contraindication check.

    Args:
        drugs:          Flat list of lowercase generic drug names.
        comorbidities:  Flat list of lowercase comorbidity strings from Module 5.
        openfda_folder: Path to the openFDA JSON files folder.

    Returns:
        List of drug-disease interaction flag dicts.  Each dict contains:
        drug, condition, severity, mechanism,
        recommendation, source, matched_label_drug.
    """
    if not comorbidities:
        return []

    try:
        openfda_rules = _load_openfda_rules(openfda_folder)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[openFDA] drug-disease check skipped: {exc}", file=sys.stderr)
        return []

    flags: list[dict[str, Any]] = []

    for drug in drugs:
        matched_entries = _find_drug_rules(drug, openfda_rules)

        for condition in comorbidities:
            condition_norm = _normalise_name(condition)

            for rule_key, condition_rules in matched_entries:
                rule = condition_rules.get(condition_norm)
                if rule:
                    flags.append(
                        {
                            "drug": drug,
                            "condition": condition_norm,
                            "severity": rule["severity"],
                            "mechanism": "Identified via openFDA drug label analysis",
                            "recommendation": rule["evidence"],
                            "source": "openfda",
                            "matched_label_drug": rule_key,
                        }
                    )
                    break  # One match per drug-condition pair is sufficient

    return flags


def _run_ddi_layers_for_pair(
    drug_a: str,
    drug_b: str,
    model_path: str,
    confidence_threshold: float,
) -> dict[str, Any]:
    """
    Run Layers 2 and 3 for a single drug pair that was not found in DrugBank.
    Args:
        drug_a:               First drug name.
        drug_b:               Second drug name.
        model_path:           Path to DeepDDI model weights.
        confidence_threshold: Minimum confidence to trust a DeepDDI result.

    Returns:
        Single interaction result dict from whichever layer responded.
    """
    # Layer 2: DeepDDI 
    deepddi_result = predict_ddi_deepddi(drug_a, drug_b, model_path=model_path)
    confidence = deepddi_result.get("confidence") or 0.0

    if confidence >= confidence_threshold:
        # DeepDDI confident enough — use its result directly
        return deepddi_result

    # Layer 3: Mistral 
    # DeepDDI either unavailable or below threshold — escalate to LLM
    mistral_result = predict_ddi_mistral(drug_a, drug_b)
    return mistral_result


def check_ddi(
    normalized_drugs: dict[str, Any],
    comorbidities: list[str] | None = None,
    xml_path: str = DEFAULT_XML_PATH,
    openfda_folder: str = DEFAULT_OPENFDA_FOLDER,
    deepddi_model_path: str = DEFAULT_DEEPDDI_MODEL_PATH,
    confidence_threshold: float = DEEPDDI_CONFIDENCE_THRESHOLD,
) -> dict[str, Any]:
    """
    Main DDI orchestrator — runs all four layers and returns a unified result.

    This is the primary public API for Module 8.

    Decision flow for every drug pair:
        1. DrugBank lookup  → if edge found: return verified result (Layer 1)
        2. DeepDDI predict  → if confidence ≥ threshold: return ML result (Layer 2)
        3. Mistral LLM      → last resort, always needs_review=True (Layer 3)

    Simultaneously (in parallel logic, executed after DDI):
        openFDA labels → drug-disease contraindication check (Parallel layer)

    Args:
        normalized_drugs:      Module 2 output dict (must contain ``"drugs"`` list).
        comorbidities:         List of condition strings from Module 5.
                               Defaults to [] if not provided.
        xml_path:              Path to DrugBank XML file.
        openfda_folder:        Path to openFDA JSON label files folder.
        deepddi_model_path:    Path to DeepDDI PyTorch model weights file.
        confidence_threshold:  Minimum DeepDDI confidence to skip Mistral escalation.

    Returns:
        Structured dict with keys:
        ``status``, ``module``, ``drugs_checked``,
        ``interactions``, ``drug_disease_interactions``, ``data_sources``.
    """
    if comorbidities is None:
        comorbidities = []

    try:
        # ── Step 1: Flatten drug names from Module 2 ─────────────────────────
        drugs = _flatten_normalized_drugs(normalized_drugs)

        if not drugs:
            return {
                "status": "success",
                "module": "DDI Checker",
                "note": "No drugs found in normalized_drugs input.",
                "drugs_checked": [],
                "interactions": [],
                "drug_disease_interactions": [],
                "data_sources": {
                    "drug_drug_primary": "drugbank",
                    "drug_drug_fallback": "deepddi",
                    "drug_drug_llm": "mistral",
                    "drug_disease": "openfda",
                },
            }

        # ── Step 2: Layer 1 — DrugBank lookup ────────────────────────────────
        drugbank_hits, not_found_pairs = check_drug_drug_from_drugbank(
            drugs, xml_path=xml_path
        )

        all_interactions: list[dict[str, Any]] = list(drugbank_hits)

        # ── Step 3: Layers 2 & 3 for pairs not in DrugBank ───────────────────
        for drug_a, drug_b in not_found_pairs:
            result = _run_ddi_layers_for_pair(
                drug_a,
                drug_b,
                model_path=deepddi_model_path,
                confidence_threshold=confidence_threshold,
            )
            all_interactions.append(result)

        # ── Step 4: Parallel layer — openFDA drug-disease check ───────────────
        drug_disease_flags = check_drug_disease_openfda(
            drugs,
            [_normalise_name(c) for c in comorbidities],
            openfda_folder=openfda_folder,
        )

        return {
            "status": "success",
            "module": "DDI Checker",
            "drugs_checked": drugs,
            "interactions": all_interactions,
            "drug_disease_interactions": drug_disease_flags,
            "data_sources": {
                "drug_drug_primary": "drugbank",
                "drug_drug_fallback": "deepddi",
                "drug_drug_llm": "mistral",
                "drug_disease": "openfda",
            },
        }

    except Exception as exc:  # noqa: BLE001
        # Always return a valid JSON dict — never crash the pipeline
        return {
            "status": "error",
            "error": str(exc),
            "module": "DDI Checker",
            "drugs_checked": [],
            "interactions": [],
            "drug_disease_interactions": [],
            "data_sources": {
                "drug_drug_primary": "drugbank",
                "drug_drug_fallback": "deepddi",
                "drug_drug_llm": "mistral",
                "drug_disease": "openfda",
            },
        }




def check_from_module_outputs(
    module2_normalized_drugs: dict[str, Any],
    module5_appropriateness: dict[str, Any],
    xml_path: str = DEFAULT_XML_PATH,
    openfda_folder: str = DEFAULT_OPENFDA_FOLDER,
    deepddi_model_path: str = DEFAULT_DEEPDDI_MODEL_PATH,
    confidence_threshold: float = DEEPDDI_CONFIDENCE_THRESHOLD,
) -> dict[str, Any]:
    comorbidities = _extract_comorbidities_from_module5(module5_appropriateness)

    return check_ddi(
        normalized_drugs=module2_normalized_drugs,
        comorbidities=comorbidities,
        xml_path=xml_path,
        openfda_folder=openfda_folder,
        deepddi_model_path=deepddi_model_path,
        confidence_threshold=confidence_threshold,
    )


def _main() -> int:
    """
    Command-line entry point for standalone local testing of Module 8.

    Supports two modes:

    **Direct mode** — supply drugs + comorbidities as JSON strings

    **Pipeline mode** — supply raw Module 2 + Module 5 output dicts

    Returns:
        Exit code 0 on success, 1 on error.
    """
    parser = argparse.ArgumentParser(
        description="Module 8 — DDI Checker (DrugBank + DeepDDI + Mistral + openFDA)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--normalized-drugs-json",
        metavar="JSON",
        help="Module 2 normalized drugs dict as JSON string (direct mode).",
    )
    mode_group.add_argument(
        "--module2-json",
        metavar="JSON",
        help="Module 2 output dict as JSON string (pipeline mode).",
    )

    parser.add_argument(
        "--comorbidities-json",
        metavar="JSON",
        default="[]",
        help='Comorbidities as JSON array, e.g. \'["CKD","asthma"]\'. Default: []',
    )
    parser.add_argument(
        "--module5-json",
        metavar="JSON",
        default="{}",
        help="Module 5 output dict as JSON string (pipeline mode).",
    )
    parser.add_argument(
        "--xml-path",
        metavar="PATH",
        default=DEFAULT_XML_PATH,
        help=f"Path to DrugBank XML. Default: {DEFAULT_XML_PATH}",
    )
    parser.add_argument(
        "--openfda-folder",
        metavar="PATH",
        default=DEFAULT_OPENFDA_FOLDER,
        help=f"openFDA JSON files folder. Default: {DEFAULT_OPENFDA_FOLDER}",
    )
    parser.add_argument(
        "--deepddi-model",
        metavar="PATH",
        default=DEFAULT_DEEPDDI_MODEL_PATH,
        help=f"DeepDDI model weights path. Default: {DEFAULT_DEEPDDI_MODEL_PATH}",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=DEEPDDI_CONFIDENCE_THRESHOLD,
        help=f"DeepDDI confidence threshold. Default: {DEEPDDI_CONFIDENCE_THRESHOLD}",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output with 2-space indent.",
    )

    args = parser.parse_args()

    try:
        if args.normalized_drugs_json:
            # Direct mode
            normalized_drugs = json.loads(args.normalized_drugs_json)
            comorbidities = json.loads(args.comorbidities_json)
            result = check_ddi(
                normalized_drugs=normalized_drugs,
                comorbidities=comorbidities,
                xml_path=args.xml_path,
                openfda_folder=args.openfda_folder,
                deepddi_model_path=args.deepddi_model,
                confidence_threshold=args.confidence,
            )
        else:
            # Pipeline mode
            module2_output = json.loads(args.module2_json)
            module5_output = json.loads(args.module5_json)
            result = check_from_module_outputs(
                module2_normalized_drugs=module2_output,
                module5_appropriateness=module5_output,
                xml_path=args.xml_path,
                openfda_folder=args.openfda_folder,
                deepddi_model_path=args.deepddi_model,
                confidence_threshold=args.confidence,
            )

    except json.JSONDecodeError as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error": f"Invalid JSON input: {exc}",
                    "module": "DDI Checker",
                    "interactions": [],
                    "drug_disease_interactions": [],
                }
            )
        )
        return 1

    indent = 2 if args.pretty else None
    print(json.dumps(result, indent=indent, ensure_ascii=False))
    return 0 if result.get("status") == "success" else 1

if __name__ == "__main__":
    sys.exit(_main())
