# ⚖️ LegiWise AI – Indian Legal Assistant Model

A fine-tuned **Large Language Model (LLM)** specialized for **Indian legal documents and queries**.  
This model powers [LegiWise AI](https://github.com/koustavx08/legiwise.ai), a local-first legal assistant for legal research, document analysis, and drafting.

---

## 📌 Overview

- Understands **Indian legal language, statutes, and procedures**  
- Analyzes and summarizes legal documents (contracts, agreements, notices)  
- Extracts and explains **key clauses** (e.g., jurisdiction, parties, governing law)  
- Provides **easy-to-understand explanations** of complex clauses  
- Assists in drafting common legal documents (NDA, Rental Agreement, Power of Attorney, etc.)  

Runs fully **offline with Ollama**, ensuring sensitive legal data remains private.

---

## ⚙️ Model Details

- **Format:** GGUF (optimized for Ollama)  
- **Size:** ~8.54 GB  
- **Base Architecture:** LLaMA/Mistral variant  
- **Embedding Model (for RAG):** `sentence-transformers/all-MiniLM-L6-v2`  
- **Dataset:** Indian legal texts, contracts, case law, FAQs  

---

## 🚀 Installation

### 1. Download the Model
Download the GGUF file from [Hugging Face](https://huggingface.co/varma007ut/Indian_Legal_Assitant) and place it in:

```bash
LEGIWISE/models/outpt_file.gguf
```

### 2. Create a Modelfile

In the root directory, create a file named `Modelfile`:

```text
FROM ./models/outpt_file.gguf
```

### 3. Register in Ollama

Make sure Ollama is running, then register the model:
```
ollama create indian-legal-assistant -f Modelfile
```

Verify installation:

```bash
ollama list
```

You should see:

```
indian-legal-assistant:latest
```

----------

## 🧑‍💻 Usage

Run directly with Ollama:

```bash
ollama run indian-legal-assistant
```

Or integrate with **LegiWise AI**:

```bash
python app.py
```

This launches a Gradio interface to:

-   Upload legal documents (PDF, DOCX, Images)
    
-   Ask document-specific questions
    
-   Summarize or analyze clauses
    
-   Draft legal templates
    

----------

## 📂 Repository Structure

```
models/              # Store GGUF model files here
Modelfile            # Model configuration for Ollama
README.md            # Documentation (this file)
```
----------

## 🔒 Privacy & Disclaimer

-   Runs **entirely offline** on your local machine
    
-   No data is sent to external servers
    
-   Safe for sensitive legal documents
    

⚠️ **Disclaimer:** This tool is for **informational and research purposes only**.  
It does **not provide legal advice** and should not replace consultation with a qualified lawyer.

----------

## 📜 License

Released under the **MIT License**.

----------

## 🌟 Acknowledgements

-   [Hugging Face – varma007ut/Indian_Legal_Assitant](https://huggingface.co/varma007ut/Indian_Legal_Assitant)
    
-   [Ollama](https://ollama.ai/)
    
-   [FAISS](https://faiss.ai/)
    
-   [LangChain](https://www.langchain.com/)
    
