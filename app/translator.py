import logging
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

logging.basicConfig(level=logging.INFO)

# Use a smaller and faster translation model
MODEL_NAME = "facebook/nllb-200-distilled-600M"

tokenizer = None
model = None
device = "cuda" if torch.cuda.is_available() else "cpu"

# Curated Indian languages (NLLB codes differ from MBART)
INDIAN_LANGUAGES = {
    "hin_Deva": "Hindi",
    "ben_Beng": "Bengali",
    "guj_Gujr": "Gujarati",
    "mal_Mlym": "Malayalam",
    "mar_Deva": "Marathi",
    "tam_Taml": "Tamil",
    "tel_Telu": "Telugu",
    "urd_Arab": "Urdu",
}

def load_model():
    global model, tokenizer
    if model is None or tokenizer is None:
        logging.info(f"Loading translation model: {MODEL_NAME} on {device}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
        model.eval()
    return model, tokenizer


def translate_text(text: str, src_lang: str, target_lang: str = "eng_Latn") -> str:
    """
    Translate text from a source language to a target language.
    Uses NLLB-200 distilled (600M) for speed on CPU.
    """
    if not text.strip():
        return ""

    try:
        model, tokenizer = load_model()

        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(model.device)

        generated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.lang_code_to_id[target_lang],
            max_new_tokens=128,  # cap output length
            num_beams=2,         # faster decoding
        )

        translated_text = tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True
        )[0]

        logging.info(f"✅ Translated {src_lang} → {target_lang}")
        return translated_text

    except Exception as e:
        logging.error(f"Translation error: {e}")
        return f"Error: {e}"


# Example usage
if __name__ == "__main__":
    print("--- Testing Translations (fast mode) ---")

    hindi_text = "अदालत ने प्रतिवादी को समन जारी किया।"
    print("Hindi → English:", translate_text(hindi_text, src_lang="hin_Deva", target_lang="eng_Latn"))

    english_text = "The court issued a summons to the defendant."
    print("English → Hindi:", translate_text(english_text, src_lang="eng_Latn", target_lang="hin_Deva"))
