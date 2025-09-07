import torch
import logging
import re
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

class RagPipeline:
    def __init__(self):
        self.vector_store = None
        self.qa_chain = None
        self.tokenizer = None # Store tokenizer for later use
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

    def create_knowledge_base(self, document_text: str):
        """Creates a searchable vector store from the document text."""
        if not document_text:
            logging.warning("⚠️ Empty document text."); return
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(text=document_text)
        self.vector_store = FAISS.from_texts(texts=chunks, embedding=self.embeddings)
        logging.info(f"📚 Knowledge base created with {len(chunks)} chunks.")

    def ask(self, question: str) -> str:
        """Asks a question to the knowledge base with a robust prompt."""
        if not self.qa_chain: return "QA chain not initialized."

        prompt = f"Question: {question}\n\nAnswer:"
        
        if self.vector_store:
            docs = self.vector_store.similarity_search(query=question, k=3)
            context = "\n".join([doc.page_content for doc in docs])
            # --- CRITICAL FIX: Truncate the combined context and prompt ---
            full_prompt_text = f"Using the following context, answer the question.\n\nContext:\n{context}\n\n{prompt}"
            truncated_prompt = self._truncate_text(full_prompt_text)
            response = self.qa_chain.invoke({"input_documents": [], "question": truncated_prompt})
        else:
            response = self.qa_chain.invoke({"input_documents": [], "question": prompt})

        raw_answer = response.get("output_text", "⚠️ Could not generate a response.")
        return raw_answer.split("Answer:")[0].strip()

    def summarize(self, text: str) -> str:
        """Summarizes a document with a robust prompt."""
        if not self.qa_chain: return "Summarization model not initialized."
        
        # --- CRITICAL FIX: Truncate the document text before creating the prompt ---
        truncated_text = self._truncate_text(text, max_length=1500)
        prompt = f"Exercise: Summarize the key points of the following document in a few clear sentences.\n\nDocument:\n`{truncated_text}`\n\nSummary:"
        
        original_vector_store = self.vector_store
        self.vector_store = None
        summary = self.ask(prompt)
        self.vector_store = original_vector_store
        return summary

