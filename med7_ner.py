import spacy
import json
import re

# Load the Med7 model
try:
    nlp = spacy.load("en_core_med7_lg")
except Exception as e:
    print(f"Error: {e}")

def extract_full_patient_data(text):
    doc = nlp(text)
    
    # Initialize the structure you requested
    data = {
        "patient": {"age": None, "gender": None},
        "complaints": [],
        "diagnosis": None,
        "drugs": []
    }

    # 1. Extract Age using Regex (looking for numbers followed by 'yo' or 'year old')
    age_match = re.search(r'(\d+)\s*(?:year old|yo|y/o|year-old)', text, re.IGNORECASE)
    if age_match:
        data["patient"]["age"] = int(age_match.group(1))

    # 2. Extract Gender
    if re.search(r'\b(male|man|m)\b', text, re.IGNORECASE):
        data["patient"]["gender"] = "M"
    elif re.search(r'\b(female|woman|f)\b', text, re.IGNORECASE):
        data["patient"]["gender"] = "F"

    # 3. Extract Complaints & Diagnosis (Heuristic approach)
    # Usually, keywords like 'fever', 'cough' are complaints. 
    # Keywords after 'diagnosed with' are the diagnosis.
    complaint_keywords = ["fever", "cough", "headache", "pain", "nausea"]
    for word in complaint_keywords:
        if word in text.lower():
            data["complaints"].append(word)

    diag_match = re.search(r'(?:diagnosed with|diagnosis of)\s+([a-zA-Z]+)', text, re.IGNORECASE)
    if diag_match:
        data["diagnosis"] = diag_match.group(1)

    # 4. Med7 Logic for Drugs
    current_drug = {}
    for ent in doc.ents:
        label = ent.label_
        if label == "DRUG":
            if current_drug: data["drugs"].append(current_drug)
            current_drug = {"name": ent.text}
        elif label in ["STRENGTH", "DOSAGE"]:
            current_drug["dose"] = ent.text
        elif label == "ROUTE":
            current_drug["route"] = ent.text
        elif label == "FREQUENCY":
            current_drug["frequency"] = ent.text

    if current_drug:
        data["drugs"].append(current_drug)

    return data

# --- TEST BLOCK ---
if __name__ == "__main__":
    sample_text = "A 45year old male presenting with fever and cough is diagnosed with pneumonia. Prescribe Amoxicillin 500mg oral TID."
    
    final_output = extract_full_patient_data(sample_text)
    print(json.dumps(final_output, indent=2))