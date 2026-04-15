# backend/modules/module5/test_module5.py
'''Used for testing Module 5'''
import json
from backend.modules.module5.drug_appropriateness import check_drug_appropriateness

def run_test_case():
    # 🔹 Sample Input (simulate upstream modules)
    input_data = {
    "patient": {"age": 45, "gender": "M"},
    "diagnosis": "migraine",
    "drugs": [
        {
            "original": "paracetamol",
            "corrected_name": "paracetamol",
            "generic_name": "paracetamol",
            "structured": {
                "dose": "500mg",
                "route": "oral",
                "frequency": "TID",
                "duration": "5 days"
            }
        }
    ]
}

    # 🔹 Call Module 5
    result = check_drug_appropriateness(
        normalized_drugs={"drugs": input_data["drugs"]},
        diagnosis=input_data["diagnosis"]
    )

    # 🔹 Pretty Print Output
    print("\n=== MODULE 5 OUTPUT ===")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    run_test_case()