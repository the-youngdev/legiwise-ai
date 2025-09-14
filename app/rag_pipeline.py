import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains.question_answering import load_qa_chain

logging.basicConfig(level=logging.INFO)

class RagPipeline:
    def __init__(self, model: str = "mistral"):
        self.vector_store = None
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        logging.info(f"Initializing Ollama with model: {model}")
        llm = Ollama(model=model, temperature=0)
        self.qa_chain = load_qa_chain(llm=llm, chain_type="stuff")
        logging.info("✅ RAG Pipeline is ready.")

    def create_knowledge_base(self, text: str):
        if not text:
            logging.warning("⚠️ Document text is empty. Knowledge base not created.")
            return
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_text(text=text)
        self.vector_store = FAISS.from_texts(texts=chunks, embedding=self.embeddings)
        logging.info(f"📚 Knowledge base created with {len(chunks)} chunks.")

    def ask(self, query: str) -> str:
        if not self.vector_store:
            return "Error: Please upload a document first to create a knowledge base."
        try:
            docs = self.vector_store.similarity_search(query=query, k=4)
            response = self.qa_chain.invoke({"input_documents": docs, "question": query})
            return response.get("output_text", "Could not generate an answer.").strip()
        except Exception as e:
            logging.error(f"Ollama ASK failed: {e}")
            return f"Error: Could not generate answer. Details: Failed to connect to Ollama. Please check that Ollama is downloaded, running and accessible. https://ollama.com/download"

    def summarize(self, text: str) -> str:
        if not text:
            return "Error: Cannot summarize empty text."
        try:
            prompt = "Summarize the key points of the following document in a few clear paragraphs:"
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=100)
            docs = text_splitter.create_documents([text])
            response = self.qa_chain.invoke({"input_documents": docs, "question": prompt})
            return response.get("output_text", "Could not generate a summary.").strip()
        except Exception as e:
            logging.error(f"Ollama SUMMARIZE failed: {e}")
            return f"Error: Could not generate summary. Details: Failed to connect to Ollama. Please check that Ollama is downloaded, running and accessible. https://ollama.com/download"