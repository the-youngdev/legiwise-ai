import torch
import logging
import re
from typing import Dict, Optional, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Configure logging
logging.basicConfig(level=logging.INFO)

# --- MODELS ---
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GENERATIVE_MODEL = "microsoft/phi-1_5"

class EnhancedRagPipeline:
    def __init__(self):
        self.vector_store = None
        self.qa_chain = None
        self.tokenizer = None
        self.document_quality_score = None  # Store quality score for adaptive prompting
        self.original_document_text = None  # Store original for reference
        self._initialize_models()

    def _initialize_models(self):
        """Initializes embeddings + generative model."""
        try:
            logging.info(f"Loading embedding model: {EMBEDDING_MODEL}")
            self.embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL,
                model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
            )
            logging.info("✅ Embedding model loaded.")

            logging.info(f"Loading generative model: {GENERATIVE_MODEL}")
            self.tokenizer = AutoTokenizer.from_pretrained(GENERATIVE_MODEL, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                GENERATIVE_MODEL,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
            pipe = pipeline(
                "text-generation", model=model, tokenizer=self.tokenizer, max_new_tokens=512,
                temperature=0.1, top_p=0.95, do_sample=True, return_full_text=False
            )
            llm = HuggingFacePipeline(pipeline=pipe)
            self.qa_chain = load_qa_chain(llm=llm, chain_type="stuff")
            logging.info("✅ Generative model + QA chain ready.")
        except Exception as e:
            logging.error(f"❌ Error initializing models: {e}")
            raise RuntimeError(f"Models could not be initialized. Error: {e}")

    def _truncate_text(self, text: str, max_length: int = 1500) -> str:
        """Truncates text to a maximum token length to prevent model errors."""
        if not self.tokenizer:
            return text[:max_length*4] # Fallback if tokenizer is not ready
        
        tokens = self.tokenizer.encode(text)
        if len(tokens) > max_length:
            truncated_tokens = tokens[:max_length]
            return self.tokenizer.decode(truncated_tokens, skip_special_tokens=True)
        return text

    def _create_quality_aware_context_prefix(self, quality_score: Optional[float] = None) -> str:
        """Creates context-aware instructions based on text quality."""
        if quality_score is None:
            quality_score = self.document_quality_score or 0.5
        
        if quality_score < 0.3:
            return """IMPORTANT: The source document contains significant OCR errors. Focus on understanding the general meaning rather than exact wording. Use contextual reasoning to interpret unclear passages. If specific details are unclear due to text quality, indicate this in your response."""
        elif quality_score < 0.7:
            return """NOTE: The source document may contain OCR recognition errors. Interpret the content contextually and focus on the overall meaning. Use your best judgment to understand potentially unclear passages."""
        else:
            return """The source document was processed via OCR and may contain minor text artifacts. Process normally while being aware of potential recognition errors."""

    def create_knowledge_base(self, document_text: str, quality_score: Optional[float] = None):
        """Creates a searchable vector store from the document text with quality awareness."""
        if not document_text:
            logging.warning("⚠️ Empty document text."); return
        
        # Store quality information
        self.document_quality_score = quality_score
        self.original_document_text = document_text
        
        # Log quality information
        if quality_score:
            quality_level = "Low" if quality_score < 0.3 else "Medium" if quality_score < 0.7 else "High"
            logging.info(f"📊 Document quality score: {quality_score:.2f} ({quality_level})")
        
        # Create chunks with quality-aware splitting
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200,
            # Add more conservative splitting for low-quality text
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""] if quality_score and quality_score < 0.5 else None
        )
        chunks = text_splitter.split_text(text=document_text)
        
        # Filter out very short or noisy chunks for low-quality text
        if quality_score and quality_score < 0.5:
            original_count = len(chunks)
            chunks = [chunk for chunk in chunks if len(chunk.strip()) > 20 and self._is_meaningful_chunk(chunk)]
            if len(chunks) < original_count:
                logging.info(f"🧹 Filtered {original_count - len(chunks)} low-quality chunks")
        
        self.vector_store = FAISS.from_texts(texts=chunks, embedding=self.embeddings)
        logging.info(f"📚 Knowledge base created with {len(chunks)} chunks.")

    def _is_meaningful_chunk(self, chunk: str) -> bool:
        """Determines if a chunk contains meaningful content (not just OCR artifacts)."""
        # Remove whitespace and check length
        clean_chunk = chunk.strip()
        if len(clean_chunk) < 20:
            return False
        
        # Count alphabetic characters vs total
        alpha_count = sum(1 for c in clean_chunk if c.isalpha())
        if alpha_count / len(clean_chunk) < 0.5:  # Less than 50% alphabetic
            return False
        
        # Check for excessive repetition (common OCR artifact)
        words = clean_chunk.split()
        if len(set(words)) < len(words) * 0.3:  # Less than 30% unique words
            return False
        
        return True

    def _create_robust_qa_prompt(self, question: str, context: str) -> str:
        """Creates a robust Q&A prompt that handles OCR errors gracefully."""
        quality_instruction = self._create_quality_aware_context_prefix()
        
        prompt = f"""{quality_instruction}

INSTRUCTIONS:
1. Read the context carefully, focusing on the overall meaning rather than individual character accuracy
2. If you encounter unclear passages, use surrounding context to infer the likely meaning
3. Answer based on what you can reasonably understand from the text
4. If the text quality makes specific details unclear, acknowledge this limitation
5. Provide the most helpful answer possible given the available information

CONTEXT:
{context}

QUESTION: {question}

ANSWER: Please provide a clear, helpful response based on the context above."""
        
        return prompt

    def _create_robust_summary_prompt(self, text: str) -> str:
        """Creates a summary prompt optimized for OCR text."""
        quality_instruction = self._create_quality_aware_context_prefix()
        
        if self.document_quality_score and self.document_quality_score < 0.3:
            instruction_details = """Focus on identifying the main topics, key points, and overall structure rather than specific details that may be corrupted by OCR errors. Ignore obvious OCR artifacts and organize the information logically."""
        else:
            instruction_details = """Summarize the key points and main ideas of the document. Focus on the essential information while being aware of potential text artifacts."""
        
        prompt = f"""{quality_instruction}

TASK: {instruction_details}

APPROACH:
1. Identify the document type and main subject matter
2. Extract key themes and important points
3. Ignore obvious OCR artifacts and focus on meaningful content
4. If certain sections are unclear due to text errors, note this in your summary
5. Organize information logically, even if the source text structure is imperfect

DOCUMENT TEXT:
{text}

SUMMARY: Provide a comprehensive summary that captures the essential information and main points of this document."""
        
        return prompt

    def ask(self, question: str) -> str:
        """Asks a question to the knowledge base with OCR-aware robust prompting."""
        if not self.qa_chain: 
            return "QA chain not initialized."
        
        if self.vector_store:
            # Retrieve relevant documents
            docs = self.vector_store.similarity_search(query=question, k=3)
            context = "\n".join([doc.page_content for doc in docs])
            
            # Create robust prompt
            full_prompt = self._create_robust_qa_prompt(question, context)
            
            # Truncate if necessary
            truncated_prompt = self._truncate_text(full_prompt)
            
            # Add quality context if document quality is poor
            if self.document_quality_score and self.document_quality_score < 0.5:
                logging.info(f"🔍 Using OCR-aware prompting (quality: {self.document_quality_score:.2f})")
            
            response = self.qa_chain.invoke({"input_documents": [], "question": truncated_prompt})
        else:
            # No knowledge base - direct question
            basic_prompt = f"Question: {question}\n\nAnswer:"
            response = self.qa_chain.invoke({"input_documents": [], "question": basic_prompt})

        raw_answer = response.get("output_text", "⚠️ Could not generate a response.")
        
        # Clean up the response
        if "Answer:" in raw_answer:
            clean_answer = raw_answer.split("Answer:")[-1].strip()
        else:
            clean_answer = raw_answer.strip()
        
        return clean_answer

    def summarize(self, text: str = None) -> str:
        """Summarizes a document with OCR-aware robust prompting."""
        if not self.qa_chain: 
            return "Summarization model not initialized."
        
        # Use provided text or stored document text
        text_to_summarize = text or self.original_document_text
        if not text_to_summarize:
            return "No document text available for summarization."
        
        # Truncate text to manageable size
        truncated_text = self._truncate_text(text_to_summarize, max_length=1200)
        
        # Create robust summary prompt
        prompt = self._create_robust_summary_prompt(truncated_text)
        final_prompt = self._truncate_text(prompt)
        
        # Log quality awareness
        if self.document_quality_score and self.document_quality_score < 0.5:
            logging.info(f"📝 Using OCR-aware summarization (quality: {self.document_quality_score:.2f})")
        
        # Temporarily disable vector store for summarization
        original_vector_store = self.vector_store
        self.vector_store = None
        
        try:
            response = self.qa_chain.invoke({"input_documents": [], "question": final_prompt})
            raw_summary = response.get("output_text", "⚠️ Could not generate a summary.")
            
            # Clean up the response
            if "SUMMARY:" in raw_summary:
                clean_summary = raw_summary.split("SUMMARY:")[-1].strip()
            elif "Summary:" in raw_summary:
                clean_summary = raw_summary.split("Summary:")[-1].strip()
            else:
                clean_summary = raw_summary.strip()
                
        finally:
            # Restore vector store
            self.vector_store = original_vector_store
        
        return clean_summary

    def get_quality_report(self) -> Dict[str, any]:
        """Returns information about the document quality and processing status."""
        return {
            "quality_score": self.document_quality_score,
            "quality_level": (
                "Low" if self.document_quality_score and self.document_quality_score < 0.3 else
                "Medium" if self.document_quality_score and self.document_quality_score < 0.7 else
                "High" if self.document_quality_score else "Unknown"
            ),
            "has_knowledge_base": self.vector_store is not None,
            "has_original_text": self.original_document_text is not None,
            "using_ocr_aware_prompts": self.document_quality_score is not None and self.document_quality_score < 0.7
        }

    def search_similar_chunks(self, query: str, k: int = 5) -> list:
        """Search for similar text chunks with quality awareness."""
        if not self.vector_store:
            return []
        
        docs = self.vector_store.similarity_search(query=query, k=k)
        results = []
        
        for doc in docs:
            # Add quality context to each result
            result = {
                "content": doc.page_content,
                "metadata": getattr(doc, 'metadata', {}),
                "quality_note": None
            }
            
            # Add quality warning for low-quality documents
            if self.document_quality_score and self.document_quality_score < 0.5:
                result["quality_note"] = "This content may contain OCR errors. Interpret contextually."
            
            results.append(result)
        
        return results

# Backward compatibility alias
RagPipeline = EnhancedRagPipeline

# Example usage and testing
def test_enhanced_pipeline():
    """Test the enhanced pipeline with sample OCR text."""
    
    # Sample OCR text with common errors
    sample_ocr_text = """
    C0mpany Pr0f1t Rep0rt 2O23
    
    Th3 c0mpany h4s sh0wn str0ng gr0wth 1n 2O23 w1th t0t4l r3v3nu3s 
    0f $5.2 m1ll10n, r3pr3s3nt1ng 4 15% 1ncr34s3 fr0m th3 pr3v10us y34r.
    
    K3y 4ch13v3m3nts 1nclud3:
    - N3w pr0duct l4unch1ng 1n Q2
    - 3xp4ns10n t0 3 n3w m4rk3ts  
    - 25% 1ncr34s3 1n cust0m3r s4t1sf4ct10n
    
    Th3 0utl00k f0r 2O24 r3m41ns p0s1t1v3 w1th pr0j3ct3d gr0wth 0f 2O%.
    """
    
    # Initialize pipeline
    print("Initializing Enhanced RAG Pipeline...")
    pipeline = EnhancedRagPipeline()
    
    # Create knowledge base with quality score
    quality_score = 0.4  # Simulated low-medium quality
    pipeline.create_knowledge_base(sample_ocr_text, quality_score)
    
    # Test Q&A
    print("\n" + "="*50)
    print("Testing Q&A with OCR-aware prompting:")
    question = "What were the company's revenues in 2023?"
    answer = pipeline.ask(question)
    print(f"Q: {question}")
    print(f"A: {answer}")
    
    # Test summarization
    print("\n" + "="*50)
    print("Testing OCR-aware summarization:")
    summary = pipeline.summarize()
    print(f"Summary: {summary}")
    
    # Get quality report
    print("\n" + "="*50)
    print("Quality Report:")
    report = pipeline.get_quality_report()
    for key, value in report.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    test_enhanced_pipeline()