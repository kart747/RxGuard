# backend/modules/module5/retrieval/guideline_db.py

from typing import Dict, Any

GUIDELINE_DB: Dict[str, Dict[str, Any]] = {
    "pneumonia": {
        "aliases": [
            "pneumonia",
            "community acquired pneumonia",
            "cap"
        ],
        "first_line": ["amoxicillin", "azithromycin"],
        "alternatives": ["ceftriaxone"],
        "avoid": [],
        "notes": "Antibiotics required for bacterial pneumonia",
        "source": "WHO"
    },
    "urinary tract infection": {
        "aliases": [
            "uti",
            "urinary tract infection"
        ],
        "first_line": ["nitrofurantoin"],
        "alternatives": ["ciprofloxacin"],
        "avoid": [],
        "notes": "Avoid broad-spectrum unless necessary",
        "source": "WHO"
    },
    "viral fever": {
        "aliases": [
            "viral fever",
            "flu",
            "common cold"
        ],
        "first_line": [],
        "alternatives": [],
        "avoid": [
            "amoxicillin",
            "azithromycin",
            "ciprofloxacin",
            "ceftriaxone"
        ],
        "notes": "Antibiotics are not indicated",
        "source": "WHO"
    }
}