# In LEGIWISE/app/clauses.py
import re
import logging
from typing import List, Dict
import jsonlines

logging.basicConfig(level=logging.INFO)

# --- BLOCK-ORIENTED CLAUSE PATTERNS ---
# These patterns are designed to capture entire sections, not just lines.
CLAUSE_PATTERNS = {
    "Parties": {
        "pattern": r'(IN THE MATTER OF[\s\S]*?--VERSUS--[\s\S]*?--RESPONDENT)',
        "risk": "Low"
    },
    "Prayer/Relief Sought": {
        "pattern": r'(PRAYER[\s\S]*?(?:pass a decree|such other relief|ends of justice)\.)',
        "risk": "High"
    },
    "Jurisdiction": {
        "pattern": r'((?:That the cause of action|this Hon\'ble Court has jurisdiction)[\s\S]*?(?:within the jurisdiction|Learned Court)\.)',
        "risk": "Medium"
    },
}

def extract_clauses(full_text: str, doc_id: str = "unknown") -> List[Dict]:
    logging.info(f"Extracting clauses from: {doc_id} with block-oriented patterns.")
    found_clauses = []
    
    # Pre-process text to make regex matching easier
    processed_text = re.sub(r'\s*\n\s*', '\n', full_text) # Normalize line breaks

    for title, data in CLAUSE_PATTERNS.items():
        match = re.search(data["pattern"], processed_text, re.IGNORECASE | re.DOTALL)
        if match:
            # Extract the entire matched block
            extracted_text = match.group(1).strip()
            found_clauses.append({
                "title": title, 
                "text": extracted_text, 
                "risk": data["risk"]
            })

    with jsonlines.open("clauses.jsonl", mode="a") as writer:
        writer.write({"doc_id": doc_id, "clauses": found_clauses})

    if not found_clauses:
        return [{"title": "Info", "text": "No specific clauses were automatically identified with the new patterns.", "risk": "N/A"}]
        
    return found_clauses