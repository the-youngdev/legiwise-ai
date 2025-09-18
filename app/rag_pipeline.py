import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains.question_answering import load_qa_chain

logging.basicConfig(level=logging.INFO)

class RagPipeline:
    def __init__(self, model: str = "phi3:mini"):  # lighter Ollama model
        self.vector_store = None

        # Faster + better embeddings for CPU
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en",       # smaller + optimized
            model_kwargs={"device": "cpu"},
            encode_kwargs={"batch_size": 64}     # batch encode → much faster
        )

        logging.info(f"Initializing Ollama with model: {model}")
        self.llm = Ollama(
            model=model,
            temperature=0,
            num_predict=256  # cap output length → prevents long slow answers
        )
        self.qa_chain = load_qa_chain(llm=self.llm, chain_type="stuff")
        logging.info("✅ Optimized RAG Pipeline is ready.")

    def create_knowledge_base(self, text: str):
        if not text:
            logging.warning("⚠️ Empty document. Knowledge base not created.")
            return

        # Larger chunks → fewer splits → faster embedding
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
        chunks = text_splitter.split_text(text=text)
        self.vector_store = FAISS.from_texts(texts=chunks, embedding=self.embeddings)
        logging.info(f"📚 Knowledge base created with {len(chunks)} chunks.")

    def ask(self, query: str) -> str:
        if not self.vector_store:
            return "Error: Please upload a document first to create a knowledge base."
        try:
            docs = self.vector_store.similarity_search(query=query, k=3)  # fewer docs = faster
            response = self.qa_chain.invoke({"input_documents": docs, "question": query})
            return response.get("output_text", "Could not generate an answer.").strip()
        except Exception as e:
            logging.error(f"Ollama ASK failed: {e}")
            return (
                "Error: Could not generate answer. "
                "Please ensure Ollama is installed and running: https://ollama.com/download"
            )

    def general_ask(self, query: str) -> str:
        try:
            prompt = (
                "You are a helpful legal assistant. "
                "Please answer the following question clearly and concisely.\n\n"
                f"Question: {query}\n\nAnswer:"
            )
            response = self.qa_chain.invoke({"input_documents": [], "question": prompt})
            return response.get("output_text", "I am unable to answer that question right now.").strip()
        except Exception as e:
            logging.error(f"Ollama GENERAL ASK failed: {e}")
            return "Error: Ollama model did not respond. Ensure it is running."

    def summarize(self, text: str) -> str:
        if not text:
            return "Error: Cannot summarize empty text."
        try:
            prompt = "Summarize the key points of the following document in 3–4 bullet points:"
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=200)
            docs = text_splitter.create_documents([text])
            response = self.qa_chain.invoke({"input_documents": docs, "question": prompt})
            return response.get("output_text", "Could not generate a summary.").strip()
        except Exception as e:
            logging.error(f"Ollama SUMMARIZE failed: {e}")
            return "Error: Ollama summarization failed. Ensure it is running."
