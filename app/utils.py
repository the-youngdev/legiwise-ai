import pypdf
import docx
import logging
from app import ocr
from pdf2image import convert_from_path
import tempfile
import os
import re

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

POPPLER_PATH = r"C:\Release-24.02.0-0\poppler-24.02.0\Library\bin"

def _post_process_ocr_text(text: str) -> str:
    """
    Cleans and restructures text from an OCR process to make it more readable
    for both humans and AI models.
    """
    if not text:
        return ""
    
    # Remove common OCR artifacts
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        # Filter out scanner watermarks and very short, likely irrelevant lines
        if "scanned with" in line.lower() or len(line.strip()) < 4:
            continue
        cleaned_lines.append(line.strip())
        
    # Join lines into a single block of text, then intelligently split into paragraphs
    full_text = " ".join(cleaned_lines)
    # Replace multiple spaces with a single space
    full_text = re.sub(r'\s+', ' ', full_text)
    # Create paragraph breaks after sentences ending with punctuation.
    full_text = re.sub(r'(?<=[.!?])\s', r'\1\n\n', full_text)
    
    logging.info("Successfully post-processed OCR text.")
    return full_text

def _extract_text_from_scanned_pdf(file_path: str) -> str:
    """Performs OCR on a scanned PDF and cleans the output."""
    logging.info("PDF appears to be scanned. Starting OCR process.")
    try:
        images = convert_from_path(file_path, poppler_path=POPPLER_PATH)
        full_text = ""
        with tempfile.TemporaryDirectory() as temp_dir:
            for i, image in enumerate(images):
                temp_image_path = os.path.join(temp_dir, f"page_{i}.jpg")
                image.save(temp_image_path, "JPEG")
                full_text += ocr.extract_text_from_image(temp_image_path) + "\n"
        
        # --- CRITICAL FIX: Clean the raw OCR output ---
        return _post_process_ocr_text(full_text)

    except Exception as e:
        logging.error(f"Error during OCR of PDF {file_path}: {e}")
        return "Error: Could not perform OCR on the PDF."

def _extract_text_from_pdf(file_path: str) -> str:
    """Extracts text from a PDF, with a fallback to OCR for scanned documents."""
    text = ""
    try:
        with open(file_path, "rb") as pdf_file:
            reader = pypdf.PdfReader(pdf_file)
            for page in reader.pages:
                text += page.extract_text() or ""
    except Exception as e:
        logging.error(f"Error reading PDF {file_path}: {e}")
        return _extract_text_from_scanned_pdf(file_path)

    if len(text.strip()) < 100:
        logging.warning("Very little text extracted from PDF. Attempting OCR as a fallback.")
        return _extract_text_from_scanned_pdf(file_path)
    
    return text

def _extract_text_from_docx(file_path: str) -> str:
    """Extracts text from a DOCX file."""
    try:
        doc = docx.Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        logging.error(f"Error reading DOCX {file_path}: {e}")
        return "Error: Could not read the DOCX file."

def _extract_text_from_txt(file_path: str) -> str:
    """Extracts text from a TXT file."""
    try:
        with open(file_path, "r", encoding="utf-8") as txt_file:
            return txt_file.read()
    except Exception as e:
        logging.error(f"Error reading TXT {file_path}: {e}")
        return "Error: Could not read the TXT file."

def load_file_content(file_path: str) -> str:
    """Loads content from a file by routing it to the correct parser."""
    if not file_path:
        return "Error: No file path provided."
    file_extension = file_path.lower().split('.')[-1]
    logging.info(f"Loading file: {file_path} with extension: {file_extension}")
    if file_extension == 'pdf':
        return _extract_text_from_pdf(file_path)
    elif file_extension == 'docx':
        return _extract_text_from_docx(file_path)
    elif file_extension == 'txt':
        return _extract_text_from_txt(file_path)
    elif file_extension in ['png', 'jpg', 'jpeg', 'bmp']:
        raw_ocr_text = ocr.extract_text_from_image(file_path)
        return _post_process_ocr_text(raw_ocr_text) # Also clean text from single images
    else:
        logging.warning(f"Unsupported file type: {file_extension}")
        return "Error: Unsupported file type."

