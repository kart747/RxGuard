#!/usr/bin/env python3
"""
Module 2: Drug normalization from OCR corrected text.

Features:
- Extract likely medication mentions from free-form prescription text.
- Fix common OCR spelling errors via fuzzy matching.
- Map brand names to generic names (local map + optional RxNorm API lookup).
- Return normalized drug list as JSON-serializable dict.

This module is dependency-light and uses only Python stdlib.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from difflib import get_close_matches
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import urlopen

# Curated high-frequency generic names used for fuzzy correction.
GENERIC_LEXICON: tuple[str, ...] = (
    "amoxicillin",
    "clavulanate",
    "azithromycin",
    "amoxicillin clavulanate",
    "paracetamol",
    "acetaminophen",
    "ibuprofen",
    "diclofenac",
    "pantoprazole",
    "omeprazole",
    "rabeprazole",
    "domperidone",
    "ondansetron",
    "cetirizine",
    "levocetirizine",
    "metformin",
    "atorvastatin",
    "rosuvastatin",
    "amlodipine",
    "telmisartan",
    "losartan",
    "clopidogrel",
    "aspirin",
    "doxycycline",
    "cefixime",
    "cefpodoxime",
    "linezolid",
)

# Local brand map for fast offline use; RxNorm lookup can refine this online.
BRAND_TO_GENERIC_LOCAL: dict[str, str] = {
    "augmentin": "amoxicillin clavulanate",
    "pan": "pantoprazole",
    "pantod": "pantoprazole",
    "enzoflam": "diclofenac",
    "dolo": "paracetamol",
    "crocin": "paracetamol",
    "calpol": "paracetamol",
    "allegra": "fexofenadine",
    "atorva": "atorvastatin",
    "glycomet": "metformin",
    "metrogyl": "metronidazole",
}

MED_PREFIXES = (
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
)


@dataclass
class DrugCandidate:
    original: str
    token: str


_RXNORM_CACHE: dict[str, dict[str, Any]] = {}

_ROUTE_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bpo\b|\boral\b|\bby mouth\b", "PO"),
    (r"\biv\b|\bintravenous\b", "IV"),
    (r"\bim\b|\bintramuscular\b", "IM"),
    (r"\bsc\b|\bsubcutaneous\b", "SC"),
    (r"\btopical\b|\bapply\b", "TOPICAL"),
)

_FREQ_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bbid\b|\bbd\b|\b1-0-1\b", "BID"),
    (r"\btid\b|\b1-1-1\b", "TID"),
    (r"\bqid\b|\b1-1-1-1\b", "QID"),
    (r"\bqd\b|\bod\b|\b1-0-0\b", "QD"),
    (r"\bhs\b", "HS"),
    (r"\bsos\b", "SOS"),
)


def _looks_like_noise(token: str) -> bool:
    if not token:
        return True
    if len(token) < 3:
        return True
    if token.isdigit():
        return True
    if re.fullmatch(r"[\W_]+", token):
        return True
    return False


def _cleanup_token(raw: str) -> str:
    token = raw.strip()
    token = re.sub(r"^[^A-Za-z]+|[^A-Za-z0-9+\-]+$", "", token)
    token = token.replace("|", "")
    token = re.sub(r"\s+", " ", token).strip()
    return token


def _extract_candidates(corrected_text: str) -> list[DrugCandidate]:
    text = corrected_text or ""
    candidates: list[DrugCandidate] = []

    # Pattern 1: medicine prefixes (Tab./Cap./Inj.)
    prefix_pattern = re.compile(
        r"\b(?:tab(?:let)?|cap(?:sule)?|inj(?:ection)?|syp|syrup|cream|drop|ointment)\.?\s*[:\-]?\s*([A-Za-z][A-Za-z0-9+\-/]{2,})",
        flags=re.IGNORECASE,
    )
    for m in prefix_pattern.finditer(text):
        token = _cleanup_token(m.group(1))
        if not _looks_like_noise(token):
            candidates.append(DrugCandidate(original=m.group(1), token=token))

    # Pattern 2: Rx block words that resemble medicine names.
    rx_match = re.search(r"\bRx\b[:,\-]?\s*(.+?)(?:\bAdv\b|$)", text, flags=re.IGNORECASE)
    if rx_match:
        rx_segment = rx_match.group(1)
        for w in re.split(r"[;,\n]|\s{2,}", rx_segment):
            w = _cleanup_token(w)
            if _looks_like_noise(w):
                continue
            if " " in w:
                # Multi-word spans from noisy OCR are usually not drug names.
                continue
            wl = w.lower()
            if wl in MED_PREFIXES:
                continue
            if re.search(r"\d", wl) and not re.search(r"[a-z]", wl):
                continue
            if wl in {"after", "before", "meals", "meal", "days", "day", "x"}:
                continue
            candidates.append(DrugCandidate(original=w, token=w))

    # Deduplicate while preserving order.
    seen: set[str] = set()
    unique: list[DrugCandidate] = []
    for c in candidates:
        key = c.token.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(c)
    return unique


def _fuzzy_correct_name(token: str) -> tuple[str, str]:
    lowered = token.lower()

    if lowered in BRAND_TO_GENERIC_LOCAL:
        return token, "brand_local"

    lexicon = set(GENERIC_LEXICON) | set(BRAND_TO_GENERIC_LOCAL.keys())
    match = get_close_matches(lowered, lexicon, n=1, cutoff=0.76)
    if not match:
        return token, "none"

    picked = match[0]
    if picked in BRAND_TO_GENERIC_LOCAL:
        return picked, "fuzzy_brand"
    return picked, "fuzzy_generic"


def _extract_context(full_text: str, original: str, corrected: str) -> str:
    text = full_text or ""
    if not text:
        return ""

    next_med_pattern = re.compile(
        r"\b(?:tab(?:let)?|cap(?:sule)?|inj(?:ection)?|syp|syrup|cream|drop|ointment)\.?\s*[:\-]?\s*[A-Za-z]",
        flags=re.IGNORECASE,
    )
    patterns = [re.escape(original), re.escape(corrected)]
    for p in patterns:
        if not p:
            continue
        m = re.search(p, text, flags=re.IGNORECASE)
        if not m:
            continue
        start = m.start()
        tail = text[m.end() :]
        next_m = next_med_pattern.search(tail)
        if next_m:
            end = m.end() + next_m.start()
        else:
            end = min(len(text), m.end() + 120)
        return text[start:end]

    return text


def _extract_dose(context: str) -> str | None:
    m = re.search(r"\b(\d+(?:\.\d+)?)\s*(mg|mcg|g|ml|iu)\b", context, flags=re.IGNORECASE)
    if not m:
        return None
    value = m.group(1)
    unit = m.group(2).lower()
    return f"{value} {unit}"


def _extract_route(context: str) -> str | None:
    for pattern, route in _ROUTE_PATTERNS:
        if re.search(pattern, context, flags=re.IGNORECASE):
            return route
    return None


def _extract_frequency(context: str) -> str | None:
    for pattern, freq in _FREQ_PATTERNS:
        if re.search(pattern, context, flags=re.IGNORECASE):
            return freq

    m = re.search(r"\b(\d-\d-\d(?:-\d)?)\b", context)
    if m:
        return m.group(1)
    return None


def _extract_duration(context: str) -> str | None:
    m = re.search(r"\bx\s*(\d{1,3})\s*(day|days|week|weeks|month|months)\b", context, flags=re.IGNORECASE)
    if m:
        return f"{m.group(1)} {m.group(2).lower()}"

    m2 = re.search(r"\bfor\s*(\d{1,3})\s*(day|days|week|weeks|month|months)\b", context, flags=re.IGNORECASE)
    if m2:
        return f"{m2.group(1)} {m2.group(2).lower()}"
    return None


def _extract_structured_fields(full_text: str, original: str, corrected: str) -> dict[str, Any]:
    context = _extract_context(full_text, original, corrected)
    return {
        "dose": _extract_dose(context),
        "route": _extract_route(context),
        "frequency": _extract_frequency(context),
        "duration": _extract_duration(context),
    }


def _rxnorm_lookup(name: str, timeout_sec: float = 3.0) -> dict[str, Any]:
    key = name.lower().strip()
    if key in _RXNORM_CACHE:
        return _RXNORM_CACHE[key]

    out: dict[str, Any] = {
        "matched": False,
        "input": name,
        "rxcui": None,
        "generic": None,
        "brand": None,
        "source": "rxnorm",
    }

    base = "https://rxnav.nlm.nih.gov/REST"

    try:
        # 1) Resolve name to RxCUI.
        url = f"{base}/rxcui.json?name={quote(name)}&search=2"
        with urlopen(url, timeout=timeout_sec) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        ids = payload.get("idGroup", {}).get("rxnormId", [])
        if not ids:
            _RXNORM_CACHE[key] = out
            return out

        rxcui = str(ids[0])
        out["rxcui"] = rxcui

        # 2) Find a normalized ingredient-level generic if available.
        rel_url = f"{base}/rxcui/{quote(rxcui)}/related.json?tty=IN+PIN+MIN"
        with urlopen(rel_url, timeout=timeout_sec) as resp:
            rel_payload = json.loads(resp.read().decode("utf-8"))

        groups = rel_payload.get("relatedGroup", {}).get("conceptGroup", [])
        for group in groups:
            props = group.get("conceptProperties", [])
            if not props:
                continue
            nm = props[0].get("name")
            if nm:
                out["generic"] = str(nm).lower()
                break

        out["brand"] = name
        out["matched"] = out["generic"] is not None
        _RXNORM_CACHE[key] = out
        return out

    except (HTTPError, URLError, TimeoutError, ValueError, KeyError, OSError):
        _RXNORM_CACHE[key] = out
        return out


def normalize_drug_list(
    corrected_text: str,
    use_rxnorm: bool = True,
    timeout_sec: float = 3.0,
) -> dict[str, Any]:
    """
    Normalize drug mentions from OCR-corrected text.

    Returns JSON-serializable dict:
    {
      "drugs": [
        {
          "original": str,
          "corrected_name": str,
          "generic_name": str | null,
          "brand_name": str | null,
          "rxcui": str | null,
          "match_source": str,
          "confidence": float
        }
      ],
      "meta": {
        "count": int,
        "rxnorm_enabled": bool
      }
    }
    """
    candidates = _extract_candidates(corrected_text)
    drugs: list[dict[str, Any]] = []

    for cand in candidates:
        corrected_name, correction_source = _fuzzy_correct_name(cand.token)
        lower_corr = corrected_name.lower().strip()

        generic_name: str | None = None
        brand_name: str | None = None
        rxcui: str | None = None
        match_source = correction_source
        confidence = 0.62 if correction_source.startswith("fuzzy") else 0.50

        if lower_corr in BRAND_TO_GENERIC_LOCAL:
            brand_name = lower_corr
            generic_name = BRAND_TO_GENERIC_LOCAL[lower_corr]
            match_source = "brand_local"
            confidence = max(confidence, 0.85)

        if use_rxnorm:
            rx = _rxnorm_lookup(corrected_name, timeout_sec=timeout_sec)
            if rx.get("matched"):
                rxcui = rx.get("rxcui")
                brand_name = rx.get("brand") or brand_name
                generic_name = rx.get("generic") or generic_name
                match_source = "rxnorm"
                confidence = max(confidence, 0.92)
            elif generic_name is None and lower_corr in GENERIC_LEXICON:
                generic_name = lower_corr
                match_source = "generic_lexicon"
                confidence = max(confidence, 0.78)
        elif generic_name is None and lower_corr in GENERIC_LEXICON:
            generic_name = lower_corr
            match_source = "generic_lexicon"
            confidence = max(confidence, 0.78)

        drugs.append(
            {
                "original": cand.original,
                "corrected_name": corrected_name,
                "generic_name": generic_name,
                "brand_name": brand_name,
                "rxcui": rxcui,
                "match_source": match_source,
                "confidence": round(float(confidence), 3),
                "structured": _extract_structured_fields(
                    full_text=corrected_text,
                    original=cand.original,
                    corrected=corrected_name,
                ),
            }
        )

    return {
        "drugs": drugs,
        "meta": {
            "count": len(drugs),
            "rxnorm_enabled": bool(use_rxnorm),
        },
    }


def _main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Normalize OCR drug text")
    parser.add_argument("--text", required=True, help="corrected_text from OCR output")
    parser.add_argument(
        "--disable-rxnorm",
        action="store_true",
        help="Skip RxNorm online lookup",
    )
    parser.add_argument("--timeout", type=float, default=3.0, help="RxNorm request timeout")
    args = parser.parse_args()

    result = normalize_drug_list(
        corrected_text=args.text,
        use_rxnorm=not args.disable_rxnorm,
        timeout_sec=args.timeout,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
