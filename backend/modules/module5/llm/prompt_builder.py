# backend/modules/module5/llm/prompt_builder.py

from typing import List, Dict, Any


def build_prompt(
    diagnosis: str,
    drugs: List[Dict[str, Any]],
    context: Dict[str, Any],
) -> str:
    """
    Build a STRICT, grounded prompt for BioMistral.
    Forces structured output and prevents hallucination.
    """

    guideline = context.get("guideline", {})
    normalized_diag = context.get("normalized_diagnosis", diagnosis)

    prompt = f"""
You are a clinical decision support system.

STRICT RULES:
- Do NOT hallucinate
- Use ONLY the provided guideline
- If unsure → return "caution"
- Output MUST be valid JSON
- No explanations outside JSON

Diagnosis: {normalized_diag}

Guideline:
First line: {guideline.get("first_line", [])}
Alternatives: {guideline.get("alternatives", [])}
Avoid: {guideline.get("avoid", [])}
Notes: {guideline.get("notes", "")}

Evaluate the following drugs:
{drugs}

Return JSON in this format:
[
  {{
    "drug": "...",
    "status": "appropriate | inappropriate | caution",
    "reason": "...",
    "confidence": 0.0-1.0,
    "source": "LLM"
  }}
]
"""

    return prompt.strip()