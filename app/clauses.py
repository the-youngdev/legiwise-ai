import re
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO)

# --- EXPANDED KEYWORD DEFINITIONS ---
CLAUSE_PATTERNS = {
    "Parties": {
        "keywords": ["parties", "between", "licensor", "licensee", "landlord", "tenant", "petitioner", "respondent"],
        "risk": "Low"
    },
    "Definitions": {
        "keywords": ["definitions", "defined", "shall mean"],
        "risk": "Low"
    },
    "Payment Terms": {
        "keywords": ["payment", "fees", "invoice", "pricing", "rent", "consideration", "monetary"],
        "risk": "High"
    },
    "Term and Termination": {
        "keywords": ["term", "terminate", "termination", "cancel", "cancellation", "end this agreement", "revoke", "last will"],
        "risk": "High"
    },
    "Governing Law and Jurisdiction": {
        "keywords": ["governing law", "jurisdiction", "governed by the laws of", "courts of"],
        "risk": "Medium"
    },
    "Limitation of Liability": {
        "keywords": ["limitation of liability", "liable for", "indemnify", "indemnity", "hold harmless", "consequential damages"],
        "risk": "High"
    },
    "Confidentiality": {
        "keywords": ["confidential", "confidentiality", "non-disclosure", "proprietary information"],
        "risk": "Medium"
    },
    "Bequests and Inheritance (Wills)": {
        "keywords": ["bequeath", "devise", "give", "will and testament", "heir", "beneficiary", "executor", "my daughters", "my son"],
        "risk": "Medium"
    }
}

def clean_text(text: str) -> str:
    """A helper function to clean up text by removing excessive newlines and spaces."""
    # Replace multiple newlines with a single one, and multiple spaces with a single one.
    text = re.sub(r'\n\s*\n', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

def extract_clauses(full_text: str, rag_agent) -> list[dict]:
    """
    Extracts legal clauses using a more robust text splitting and keyword search method.
    """
    if not full_text:
        return []

    logging.info("Starting fast keyword-based clause extraction with improved text handling...")
    
    # Clean the full text first to handle OCR messiness
    cleaned_text = clean_text(full_text)
    
    # Split by single newline characters, which is more reliable for OCR output
    paragraphs = [p.strip() for p in cleaned_text.split('\n') if p.strip() and len(p) > 20]
    found_clauses = []
    found_titles = set()

    for title, data in CLAUSE_PATTERNS.items():
        if title in found_titles:
            continue

        for para in paragraphs:
            # Use a more forgiving lowercase search instead of strict regex
            para_lower = para.lower()
            if any(keyword in para_lower for keyword in data['keywords']):
                found_clauses.append({
                    "title": title,
                    "text": para,
                    "risk": data["risk"]
                })
                found_titles.add(title)
                logging.info(f"Found potential '{title}' clause.")
                break 
                
    if not found_clauses:
        logging.warning("No clauses found using the expanded keyword search.")
        return [{"title": "Info", "text": "No specific clauses were automatically identified. You can still ask questions about the document.", "risk": "Low"}]

    return found_clauses

