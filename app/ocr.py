import pytesseract
from PIL import Image
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Path to Tesseract (update this if it's not in your system's PATH)
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


def extract_text_from_image(image_path: str, lang: str = "eng") -> str:
    """
    Extracts text from an image using Tesseract OCR.
    """
    try:
        logging.info(f"🖼️ Running Tesseract OCR on {image_path} with lang={lang}")
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image, lang=lang)

        if not text.strip():
            logging.warning(f"⚠️ No text detected in {image_path}.")
            return ""

        logging.info(f"✅ Tesseract OCR completed, extracted {len(text)} characters.")
        return text.strip()

    except FileNotFoundError:
        logging.error(f"❌ OCR Error: The file was not found at {image_path}")
        return f"Error: The file was not found at {image_path}"
    except Exception as e:
        logging.error(f"❌ Tesseract OCR failed on {image_path}: {e}")
        return "Error: Tesseract OCR failed."