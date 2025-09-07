import logging
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import torch

# Configure basic logging
logging.basicConfig(level=logging.INFO)

# --- 1. INITIALIZE TRANSLATION MODEL AND TOKENIZER ---

MODEL_NAME = "facebook/mbart-large-50-many-to-many-mmt"
tokenizer = None
model = None

# A curated list of Indian languages supported by the model for easy access.
INDIAN_LANGUAGES = {
    "bn_IN": "Bengali",
    "gu_IN": "Gujarati",
    "hi_IN": "Hindi",
    "ml_IN": "Malayalam",
    "mr_IN": "Marathi",
    "ta_IN": "Tamil",
    "te_IN": "Telugu",
    "ur_PK": "Urdu" # Urdu is also widely used in India
}

# The full list of all 50 languages remains available if needed.
ALL_SUPPORTED_LANGUAGES = {
    "ar_AR": "Arabic", "cs_CZ": "Czech", "de_DE": "German", "en_XX": "English",
    "es_XX": "Spanish", "et_EE": "Estonian", "fi_FI": "Finnish", "fr_XX": "French",
    "gu_IN": "Gujarati", "hi_IN": "Hindi", "it_IT": "Italian", "ja_XX": "Japanese",
    "kk_KZ": "Kazakh", "ko_KR": "Korean", "lt_LT": "Lithuanian", "lv_LV": "Latvian",
    "my_MM": "Burmese", "ne_NP": "Nepali", "nl_XX": "Dutch", "ro_RO": "Romanian",
    "ru_RU": "Russian", "si_LK": "Sinhala", "tr_TR": "Turkish", "vi_VN": "Vietnamese",
    "zh_CN": "Chinese", "af_ZA": "Afrikaans", "az_AZ": "Azerbaijani", "bn_IN": "Bengali",
    "fa_IR": "Persian", "he_IL": "Hebrew", "hr_HR": "Croatian", "id_ID": "Indonesian",
    "ka_GE": "Georgian", "km_KH": "Khmer", "mk_MK": "Macedonian", "ml_IN": "Malayalam",
    "mn_MN": "Mongolian", "mr_IN": "Marathi", "pl_PL": "Polish", "ps_AF": "Pashto",
    "pt_XX": "Portuguese", "sv_SE": "Swedish", "sw_KE": "Swahili", "ta_IN": "Tamil",
    "te_IN": "Telugu", "th_TH": "Thai", "tl_XX": "Tagalog", "uk_UA": "Ukrainian",
    "ur_PK": "Urdu", "xh_ZA": "Xhosa", "gl_ES": "Galician", "sl_SI": "Slovene"
}

try:
    logging.info(f"Loading translation model: {MODEL_NAME}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tokenizer = MBart50TokenizerFast.from_pretrained(MODEL_NAME)
    model = MBartForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
    
    if device == "cuda":
        logging.info("Translation model loaded successfully onto GPU.")
    else:
        logging.info("Translation model loaded successfully onto CPU.")
        
except Exception as e:
    logging.error(f"Failed to initialize the translation model: {e}")

def translate_text(text: str, src_lang: str, target_lang: str = "en_XX") -> str:
    """
    Translates a text from a source language to a target language.
    """
    if not model or not tokenizer:
        return "Error: The translation service is not available."

    try:
        tokenizer.src_lang = src_lang
        encoded_text = tokenizer(text, return_tensors="pt").to(model.device)
        generated_tokens = model.generate(
            **encoded_text,
            forced_bos_token_id=tokenizer.lang_code_to_id[target_lang]
        )
        translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        logging.info(f"Successfully translated text from {src_lang} to {target_lang}.")
        return translated_text
    except Exception as e:
        logging.error(f"An error occurred during translation: {e}")
        return f"Error during translation: {str(e)}"

# Example Usage (can be run directly to test the module)
if __name__ == '__main__':
    print("--- Testing Indian Language Translations ---")

    # --- English to Hindi Example ---
    english_text = "The court issued a summons to the defendant."
    print(f"\nOriginal (English - en_XX): {english_text}")
    translated_hindi = translate_text(english_text, src_lang="en_XX", target_lang="hi_IN")
    print(f"Translated (Hindi - hi_IN): {translated_hindi}")

    # --- Hindi to English Example ---
    hindi_text = "अदालत ने प्रतिवादी को समन जारी किया।"
    print(f"\nOriginal (Hindi - hi_IN): {hindi_text}")
    translated_english = translate_text(hindi_text, src_lang="hi_IN", target_lang="en_XX")
    print(f"Translated (English - en_XX): {translated_english}")

    # --- Bengali to English Example ---
    bengali_text = "আদালত আসামিকে সমন জারি করেছে।"
    print(f"\nOriginal (Bengali - bn_IN): {bengali_text}")
    translated_english_bengali = translate_text(bengali_text, src_lang="bn_IN", target_lang="en_XX")
    print(f"Translated (English - en_XX): {translated_english_bengali}")

    # --- Marathi to English Example ---
    marathi_text = "न्यायालयाने प्रतिवादीला समन्स बजावले."
    print(f"\nOriginal (Marathi - mr_IN): {marathi_text}")
    translated_english_marathi = translate_text(marathi_text, src_lang="mr_IN", target_lang="en_XX")
    print(f"Translated (English - en_XX): {translated_english_marathi}")

    # --- Tamil to English Example ---
    tamil_text = "நீதிமன்றம் பிரதிவாதிக்கு சம்மன் அனுப்பியது."
    print(f"\nOriginal (Tamil - ta_IN): {tamil_text}")
    translated_english_tamil = translate_text(tamil_text, src_lang="ta_IN", target_lang="en_XX")
    print(f"Translated (English - en_XX): {translated_english_tamil}")

