import pypdf
import docx
import logging
from pdf2image import convert_from_path
import tempfile
import os
import re
from . import ocr

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

POPPLER_PATH = r"C:\Release-24.02.0-0\poppler-24.02.0\Library\bin"

def _post_process_ocr_text(text: str) -> str:
    """
    Cleans and restructures text from an OCR process to make it more readable.
    """
    if not text:
        return ""
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    lines = text.split('\n')
    cleaned_lines = [line.strip() for line in lines if "scanned with" not in line.lower() and len(line.strip()) > 3]
    full_text = " ".join(cleaned_lines)
    full_text = re.sub(r'\s+', ' ', full_text).strip()
    full_text = re.sub(r'(?<=[.!?])\s+', "\n\n", full_text)
    logging.info("✅ Successfully post-processed OCR text.")
    return full_text

def _extract_text_from_scanned_pdf(file_path: str) -> str:
    """Performs OCR on a scanned PDF and cleans the output."""
    logging.info("📄 PDF appears to be scanned. Starting OCR process.")
    try:
        images = convert_from_path(file_path)
        full_text = ""
        with tempfile.TemporaryDirectory() as temp_dir:
            for i, image in enumerate(images):
                temp_image_path = os.path.join(temp_dir, f"page_{i}.jpg")
                image.save(temp_image_path, "JPEG")
                full_text += ocr.extract_text_from_image(temp_image_path) + "\n"
        return _post_process_ocr_text(full_text)
    except Exception as e:
        logging.error(f"❌ Error during OCR of PDF {file_path}: {e}")
        return "Error: Could not perform OCR on the PDF."

def _extract_text_from_pdf(file_path: str) -> str:
    """Extracts text from a PDF, with a fallback to OCR for scanned documents."""
    try:
        with open(file_path, "rb") as pdf_file:
            reader = pypdf.PdfReader(pdf_file)
            text = "".join(page.extract_text() or "" for page in reader.pages)
        if len(text.strip()) < 100:
            logging.warning("⚠️ Very little text extracted. Attempting OCR fallback.")
            return _extract_text_from_scanned_pdf(file_path)
        return text
    except Exception as e:
        logging.error(f"❌ Error reading PDF {file_path}: {e}")
        return _extract_text_from_scanned_pdf(file_path)

def _extract_text_from_docx(file_path: str) -> str:
    """Extracts text from a DOCX file."""
    try:
        doc = docx.Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        logging.error(f"❌ Error reading DOCX {file_path}: {e}")
        return "Error: Could not read the DOCX file."

def _extract_text_from_txt(file_path: str) -> str:
    """Extracts text from a TXT file with encoding fallback."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        with open(file_path, "r", encoding="latin-1") as f:
            return f.read()
    except Exception as e:
        logging.error(f"❌ Error reading TXT {file_path}: {e}")
        return "Error: Could not read the TXT file."

def load_file_content(file_path: str) -> str:
    """Loads content from a file by routing it to the correct parser."""
    if not file_path:
        return "Error: No file path provided."
    ext = file_path.lower().split('.')[-1]
    logging.info(f"📂 Loading file: {file_path} (.{ext})")
    
    handlers = {
        'pdf': _extract_text_from_pdf,
        'docx': _extract_text_from_docx,
        'txt': _extract_text_from_txt,
        'png': ocr.extract_text_from_image,
        'jpg': ocr.extract_text_from_image,
        'jpeg': ocr.extract_text_from_image,
        'bmp': ocr.extract_text_from_image,
    }
    
    handler = handlers.get(ext)
    if not handler:
        return "Error: Unsupported file type."
        
    content = handler(file_path)
    return _post_process_ocr_text(content) if ext in ['png', 'jpg', 'jpeg', 'bmp'] else content