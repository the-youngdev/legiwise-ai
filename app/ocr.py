import easyocr
import torch
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def initialize_reader():
    """Initializes the EasyOCR reader, automatically selecting GPU if available."""
    use_gpu = torch.cuda.is_available()
    logging.info(f"EasyOCR will use {'GPU' if use_gpu else 'CPU'}.")
    # Initialize for English. Add other languages like ['en', 'hi'] if needed.
    return easyocr.Reader(['en'], gpu=use_gpu)

# Initialize the reader once when the module is loaded
try:
    reader = initialize_reader()
except Exception as e:
    logging.error(f"Failed to initialize EasyOCR reader: {e}")
    reader = None

def extract_text_from_image(image_path: str) -> str:
    """
    Extracts text from an image file using EasyOCR.

    Args:
        image_path: The file path to the image.

    Returns:
        A single string of the extracted text, or an error message.
    """
    if reader is None:
        return "Error: OCR reader is not initialized. Please check the logs."

    try:
        # readtext returns a list; paragraph=True helps group related text.
        result = reader.readtext(image_path, detail=0, paragraph=True)
        return "\n".join(result)
    except FileNotFoundError:
        logging.error(f"OCR Error: File not found at {image_path}")
        return f"Error: The file was not found at {image_path}"
    except Exception as e:
        logging.error(f"An unexpected OCR error occurred: {e}")
        return f"An error occurred during OCR processing: {e}"
