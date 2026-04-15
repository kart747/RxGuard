# backend/modules/module5/scoring/confidence.py

from typing import List, Dict, Any


def apply_confidence(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Normalize and enforce confidence values across outputs.
    """

    normalized: List[Dict[str, Any]] = []

    for r in results:
        confidence = r.get("confidence")

        # Fallback mapping if missing
        if confidence is None:
            source = r.get("source", "")

            if "rule" in source:
                confidence = 0.85
            elif "llm" in source:
                confidence = 0.6
            else:
                confidence = 0.5

        # Clamp values
        confidence = max(0.0, min(1.0, float(confidence)))

        normalized.append({
            "drug": r.get("drug"),
            "status": r.get("status", "caution"),
            "reason": r.get("reason", ""),
            "confidence": confidence,
            "source": r.get("source", "unknown"),
        })

    return normalized