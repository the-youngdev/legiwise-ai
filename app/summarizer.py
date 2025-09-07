# In app/summarizer.py
from transformers import pipeline
import torch
import logging

# Configure basic logging to see informational messages
logging.basicConfig(level=logging.INFO)

# --- 1. INITIALIZE THE SUMMARIZATION PIPELINE ---

def initialize_summarizer():
    """
    Initializes the Hugging Face summarization pipeline,
    automatically selecting the GPU if available.
    """
    try:
        device_id = 0 if torch.cuda.is_available() else -1
        if device_id == 0:
            logging.info("GPU found. Initializing summarizer on GPU.")
        else:
            logging.info("No GPU found. Initializing summarizer on CPU.")
            
        # Using a well-known model for summarization.
        # You can swap this with other models like 'sshleifer/distilbart-cnn-12-6'
        summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            device=device_id
        )
        return summarizer
    except Exception as e:
        logging.error(f"Failed to initialize summarizer pipeline: {e}")
        return None

# Load the model once when the module is imported
summarizer_pipeline = initialize_summarizer()

# --- 2. DEFINE THE SUMMARIZATION FUNCTION ---

def summarize_text(text: str, min_length: int = 50, max_length: int = 250) -> str:
    """
    Summarizes a given text. Handles long texts by splitting them into chunks.

    Args:
        text (str): The text to be summarized.
        min_length (int): The minimum length of the summary.
        max_length (int): The maximum length of the summary.

    Returns:
        str: The summarized text, or an error message.
    """
    if summarizer_pipeline is None:
        return "Error: Summarizer is not initialized. Please check logs."

    if not isinstance(text, str) or not text.strip():
        return "Error: Input text must be a non-empty string."

    try:
        # Most summarization models have a token limit (e.g., 1024 for BART).
        # We need to chunk the text if it's too long.
        # A simple approximation: 1 token ~= 4 characters
        max_chunk_length = 1000 * 4 
        chunks = [text[i:i + max_chunk_length] for i in range(0, len(text), max_chunk_length)]
        
        logging.info(f"Text split into {len(chunks)} chunk(s) for summarization.")

        summary_parts = []
        for chunk in chunks:
            # Perform summarization on each chunk
            summary = summarizer_pipeline(
                chunk,
                max_length=max_length // len(chunks), # Adjust max_length per chunk
                min_length=min_length // len(chunks), # Adjust min_length per chunk
                do_sample=False
            )
            summary_parts.append(summary[0]['summary_text'])
            
        return " ".join(summary_parts)

    except Exception as e:
        logging.error(f"An error occurred during summarization: {e}")
        return f"Error during summarization: {e}"