
from transformers import pipeline, AutoTokenizer
import torch
import logging
import re
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
import gc

# Configure basic logging to see informational messages
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 1. CONFIGURATION AND SETTINGS ---

@dataclass
class SummarizerConfig:
    """Configuration for the summarizer."""
    model_name: str = "facebook/bart-large-cnn"
    fallback_model: str = "sshleifer/distilbart-cnn-12-6"  # Smaller, faster fallback
    max_chunk_tokens: int = 900  # Leave buffer for special tokens
    min_summary_length: int = 50
    max_summary_length: int = 250
    overlap_sentences: int = 2  # Sentences to overlap between chunks
    use_gpu: bool = True
    batch_size: int = 1

class EnhancedSummarizer:
    """Enhanced summarizer with quality awareness and better error handling."""
    
    def __init__(self, config: Optional[SummarizerConfig] = None):
        self.config = config or SummarizerConfig()
        self.pipeline = None
        self.tokenizer = None
        self.device = None
        self.model_loaded = False
        self.initialization_error = None
        
    def _determine_device(self) -> int:
        """Determine the best device for inference."""
        if not self.config.use_gpu or not torch.cuda.is_available():
            logging.info("Using CPU for summarization")
            return -1
        
        # Check GPU memory
        try:
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9  # GB
            if gpu_memory < 4:  # Less than 4GB VRAM
                logging.warning(f"Limited GPU memory ({gpu_memory:.1f}GB), consider using CPU")
            else:
                logging.info(f"Using GPU with {gpu_memory:.1f}GB memory for summarization")
            return 0
        except Exception as e:
            logging.warning(f"Error checking GPU memory: {e}, falling back to CPU")
            return -1

    def initialize(self) -> bool:
        """
        Initialize the summarization pipeline with error handling and fallbacks.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if self.model_loaded:
            return True
            
        try:
            self.device = self._determine_device()
            
            # Try to load the primary model
            try:
                logging.info(f"Initializing summarizer with model: {self.config.model_name}")
                self.pipeline = pipeline(
                    "summarization",
                    model=self.config.model_name,
                    device=self.device,
                    torch_dtype=torch.float16 if self.device == 0 else torch.float32,
                )
                self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
                
            except Exception as primary_error:
                logging.warning(f"Primary model failed ({primary_error}), trying fallback model")
                
                # Try fallback model
                try:
                    self.pipeline = pipeline(
                        "summarization",
                        model=self.config.fallback_model,
                        device=self.device,
                        torch_dtype=torch.float16 if self.device == 0 else torch.float32,
                    )
                    self.tokenizer = AutoTokenizer.from_pretrained(self.config.fallback_model)
                    self.config.model_name = self.config.fallback_model  # Update config
                    
                except Exception as fallback_error:
                    self.initialization_error = f"Both models failed: {primary_error}, {fallback_error}"
                    logging.error(self.initialization_error)
                    return False
            
            self.model_loaded = True
            logging.info("✅ Summarizer initialized successfully")
            return True
            
        except Exception as e:
            self.initialization_error = f"Initialization failed: {e}"
            logging.error(self.initialization_error)
            return False

    def _smart_text_chunking(self, text: str, quality_score: float = 1.0) -> List[str]:
        """
        Intelligently chunk text based on sentences and token limits.
        
        Args:
            text: Input text to chunk
            quality_score: Quality score for OCR text (0-1)
            
        Returns:
            List of text chunks
        """
        if not self.tokenizer:
            # Fallback to simple chunking if tokenizer not available
            return self._fallback_chunking(text)
        
        # For low-quality text, use more conservative chunking
        if quality_score < 0.5:
            # Use paragraph-based chunking for noisy text
            paragraphs = text.split('\n\n')
            chunks = []
            current_chunk = ""
            
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                    
                test_chunk = current_chunk + "\n\n" + para if current_chunk else para
                
                # Check token count
                if len(self.tokenizer.encode(test_chunk)) <= self.config.max_chunk_tokens:
                    current_chunk = test_chunk
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                        current_chunk = para
                    else:
                        # Paragraph itself is too long, split by sentences
                        chunks.extend(self._split_long_paragraph(para))
            
            if current_chunk:
                chunks.append(current_chunk)
            
            return chunks if chunks else [text]
        
        # High-quality text: use sentence-based chunking
        return self._sentence_based_chunking(text)

    def _sentence_based_chunking(self, text: str) -> List[str]:
        """Split text into chunks based on sentences with overlap."""
        # Split into sentences (improved regex)
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return [text]
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for i, sentence in enumerate(sentences):
            sentence_tokens = len(self.tokenizer.encode(sentence))
            
            # If adding this sentence exceeds limit, finalize current chunk
            if current_tokens + sentence_tokens > self.config.max_chunk_tokens and current_chunk:
                chunks.append(' '.join(current_chunk))
                
                # Start new chunk with overlap
                overlap_start = max(0, len(current_chunk) - self.config.overlap_sentences)
                current_chunk = current_chunk[overlap_start:]
                current_tokens = sum(len(self.tokenizer.encode(s)) for s in current_chunk)
            
            current_chunk.append(sentence)
            current_tokens += sentence_tokens
        
        # Add the last chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks if chunks else [text]

    def _split_long_paragraph(self, paragraph: str) -> List[str]:
        """Split a long paragraph that exceeds token limits."""
        words = paragraph.split()
        chunks = []
        current_chunk = []
        
        for word in words:
            test_chunk = ' '.join(current_chunk + [word])
            
            if len(self.tokenizer.encode(test_chunk)) <= self.config.max_chunk_tokens:
                current_chunk.append(word)
            else:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = [word]
                else:
                    # Single word exceeds limit (very rare), just add it
                    chunks.append(word)
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks

    def _fallback_chunking(self, text: str) -> List[str]:
        """Fallback chunking when tokenizer is not available."""
        # Approximate: 4 characters per token
        max_chars = self.config.max_chunk_tokens * 4
        
        # Split on sentence boundaries when possible
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk + " " + sentence) <= max_chars:
                current_chunk = (current_chunk + " " + sentence).strip()
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks if chunks else [text[:max_chars]]

    def _create_quality_aware_prompt(self, quality_score: float) -> str:
        """Create summarization instructions based on text quality."""
        if quality_score < 0.3:
            return "Focus on the main ideas and key points. The text may contain OCR errors, so prioritize understanding the overall meaning rather than specific details."
        elif quality_score < 0.7:
            return "Summarize the key points and main ideas. Some text recognition errors may be present, so focus on the general content."
        else:
            return "Provide a comprehensive summary of the key points and important details."

    def summarize_text(
        self, 
        text: str, 
        min_length: Optional[int] = None, 
        max_length: Optional[int] = None,
        quality_score: float = 1.0
    ) -> str:
        """
        Summarize text with quality awareness and enhanced error handling.
        
        Args:
            text: Text to summarize
            min_length: Minimum summary length (optional)
            max_length: Maximum summary length (optional)
            quality_score: Quality score for OCR text (0-1, default 1.0)
            
        Returns:
            Summarized text or error message
        """
        # Initialize if not already done
        if not self.model_loaded:
            if not self.initialize():
                return f"Error: Summarizer initialization failed - {self.initialization_error}"
        
        # Validate inputs
        if not isinstance(text, str) or not text.strip():
            return "Error: Input text must be a non-empty string."
        
        # Use provided lengths or defaults
        min_len = min_length or self.config.min_summary_length
        max_len = max_length or self.config.max_summary_length
        
        # Adjust lengths based on text quality
        if quality_score < 0.5:
            # For low-quality text, be more conservative
            min_len = max(min_len - 20, 30)
            max_len = min(max_len + 50, 300)
        
        try:
            # Smart chunking based on quality
            chunks = self._smart_text_chunking(text, quality_score)
            
            if len(chunks) == 1:
                return self._summarize_single_chunk(chunks[0], min_len, max_len, quality_score)
            else:
                return self._summarize_multiple_chunks(chunks, min_len, max_len, quality_score)
                
        except Exception as e:
            logging.error(f"Summarization error: {e}")
            return f"Error during summarization: {e}"

    def _summarize_single_chunk(self, text: str, min_len: int, max_len: int, quality_score: float) -> str:
        """Summarize a single chunk of text."""
        try:
            result = self.pipeline(
                text,
                max_length=max_len,
                min_length=min_len,
                do_sample=False,
                truncation=True
            )
            summary = result[0]['summary_text']
            
            # Add quality context if needed
            if quality_score < 0.5:
                summary += f"\n\n*Note: Summary based on text with {quality_score:.0%} quality score.*"
            
            return summary
            
        except Exception as e:
            logging.error(f"Single chunk summarization failed: {e}")
            return f"Error summarizing text: {e}"

    def _summarize_multiple_chunks(self, chunks: List[str], min_len: int, max_len: int, quality_score: float) -> str:
        """Summarize multiple chunks and combine results."""
        try:
            logging.info(f"Summarizing {len(chunks)} chunks")
            
            # Adjust per-chunk limits
            chunk_max_len = max(max_len // len(chunks), 50)
            chunk_min_len = max(min_len // len(chunks), 20)
            
            summary_parts = []
            
            for i, chunk in enumerate(chunks):
                try:
                    result = self.pipeline(
                        chunk,
                        max_length=chunk_max_len,
                        min_length=chunk_min_len,
                        do_sample=False,
                        truncation=True
                    )
                    summary_parts.append(result[0]['summary_text'])
                    
                except Exception as chunk_error:
                    logging.warning(f"Failed to summarize chunk {i+1}: {chunk_error}")
                    # Try to extract key sentences as fallback
                    sentences = chunk.split('. ')
                    key_sentences = sentences[:2] if len(sentences) > 2 else sentences
                    summary_parts.append('. '.join(key_sentences))
            
            # Combine summaries
            combined_summary = ' '.join(summary_parts)
            
            # If combined summary is too long, summarize again
            if len(combined_summary.split()) > max_len:
                try:
                    final_result = self.pipeline(
                        combined_summary,
                        max_length=max_len,
                        min_length=min_len,
                        do_sample=False,
                        truncation=True
                    )
                    combined_summary = final_result[0]['summary_text']
                except:
                    # Fallback: just truncate
                    words = combined_summary.split()
                    combined_summary = ' '.join(words[:max_len])
            
            # Add quality context if needed
            if quality_score < 0.5:
                combined_summary += f"\n\n*Note: Summary based on text with {quality_score:.0%} quality score.*"
            
            return combined_summary
            
        except Exception as e:
            logging.error(f"Multi-chunk summarization failed: {e}")
            return f"Error summarizing multiple chunks: {e}"

    def cleanup(self):
        """Clean up resources."""
        if self.pipeline:
            del self.pipeline
            self.pipeline = None
        if self.tokenizer:
            del self.tokenizer  
            self.tokenizer = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        gc.collect()
        self.model_loaded = False
        logging.info("Summarizer resources cleaned up")

    def get_status(self) -> Dict[str, any]:
        """Get current status information."""
        return {
            "model_loaded": self.model_loaded,
            "model_name": self.config.model_name,
            "device": "GPU" if self.device == 0 else "CPU",
            "initialization_error": self.initialization_error
        }

# --- 2. GLOBAL INSTANCE AND CONVENIENCE FUNCTIONS ---

# Global instance (lazy initialization)
_global_summarizer = None

def get_summarizer() -> EnhancedSummarizer:
    """Get or create the global summarizer instance."""
    global _global_summarizer
    if _global_summarizer is None:
        _global_summarizer = EnhancedSummarizer()
    return _global_summarizer

def summarize_text(text: str, min_length: int = 50, max_length: int = 250, quality_score: float = 1.0) -> str:
    """
    Convenience function for text summarization with quality awareness.
    
    Args:
        text: Text to summarize
        min_length: Minimum summary length
        max_length: Maximum summary length  
        quality_score: Quality score for OCR text (0-1)
        
    Returns:
        Summarized text or error message
    """
    summarizer = get_summarizer()
    return summarizer.summarize_text(text, min_length, max_length, quality_score)

def cleanup_summarizer():
    """Clean up the global summarizer instance."""
    global _global_summarizer
    if _global_summarizer:
        _global_summarizer.cleanup()
        _global_summarizer = None

# --- 3. BACKWARD COMPATIBILITY ---

# Keep the old function signature for backward compatibility
def initialize_summarizer():
    """Legacy function - now returns success status."""
    summarizer = get_summarizer()
    return summarizer.initialize()

# Initialize on import can be enabled with this flag
INITIALIZE_ON_IMPORT = False

if INITIALIZE_ON_IMPORT:
    summarizer_pipeline = get_summarizer()
    if not summarizer_pipeline.initialize():
        summarizer_pipeline = None
        logging.error("Failed to initialize summarizer on import")
else:
    # Lazy initialization
    summarizer_pipeline = None
    logging.info("Summarizer will be initialized on first use")