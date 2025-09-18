# ⚖️ LegiWise AI: Project Status & Roadmap 
# ⚖️ LegiWise AI  



**Date:** September 7, 2025  
**Status:** Core Features Implemented · Final Debugging Phase  

---

## 1. Project Overview  
The goal of the **LegiWise AI** project is to create a comprehensive, AI-powered legal assistant.  
All major features have been built, and the application is in the final phase of **debugging and stabilization**.  

---

## 2. What We Have Built (Confirmed Working)  

- 📑 **Document Analysis & Q&A**: This is the main feature for interacting with your own legal documents.

Upload a File: Start by uploading a PDF, DOCX, or image file (.png, .jpg) in the "Document Analysis & Q&A" tab.

Processing: The app will automatically extract the text (using Tesseract OCR for images and scanned PDFs), clean it, and build a knowledge base. You'll see the status updates as it works.

Ask Questions: Once processing is complete, you can type questions about the document into the chatbox at the bottom and get answers from the AI.

Generate a Summary: Click the "Generate Summary" button to get a concise summary of the entire document.
- **Extracted Clauses**: This tab helps you quickly identify important clauses in your document.

Automatic Extraction: After you upload a document, this tab will automatically populate with any legal clauses the app identifies (e.g., Payment Terms, Termination).

AI Explanation: For each clause found, you can click the "🤖 Explain this Clause with AI" button to get a simple, easy-to-understand explanation of what it means.
- 🧭 **Legal Navigator**: This feature provides pre-written guides for common legal procedures.

Select a Topic: Go to the "Legal Navigator" tab and choose a topic from the dropdown menu (e.g., "Filing an FIR").

Ask Questions: The guide content will be loaded into the AI's context. You can then ask specific questions related to that topic in the chatbox.  
- ✍️ **Legal Drafting Assistant**: Dynamic form + document generation working as expected.  
- 🌐 **Multilingual Translation**: Underlying translation module is functional.  

---


This project combines **RAG (Retrieval-Augmented Generation)** with **Ollama** and HuggingFace models to create a secure, efficient, and modular legal research tool.  

---

## 🚀 Features  

✅ **Chatbot Q&A** – Ask questions about uploaded case files or legal documents.  
✅ **Summarization** – Summarize full legal documents or specific sections.  
❌ **Explain Clause with AI** – *(Currently under development, not functional yet)*  
⚡ **Local Inference with Ollama** – Runs on **Mistral** for lightweight GPU-friendly inference.  
📂 **Document Support** – Works with PDF, DOCX, and OCR (scanned PDFs via Tesseract).  

---

## 🛠️ Tech Stack  

- **Backend:** FastAPI + LangChain  
- **Frontend / UI:** Gradio  
- **Models:**  
  - ✅ **Ollama (Mistral)** – Current default backend  
  - 🔄 **WIP:** Integrating [`varma007ut/Indian_Legal_Assistant`](https://huggingface.co/varma007ut/Indian_Legal_Assistant) with Ollama for India-specific legal NLP  
- **Vector DB:** FAISS for fast retrieval  
- **Extras:** gTTS for TTS, Whisper for STT, pdfplumber & docx for ingestion  

---

## 5. Features  

- 📑 Document Q&A  
- 📝 Summarization  
- 📌 Clause Extraction 
- 🧭 Legal Navigator  
- ✍️ Document Drafting  
- 🎙️ Voice Input/Output *(planned)*  
- 📊 Analytics Dashboard *(planned)*  

---

## 6. Installation  

```bash
pip install -r requirements.txt
```

---

## 7. Usage  

Run the main app with:  

```bash
python app.py
```

This will start the **Gradio web interface** for LegiWise AI.  
Once running, open the provided `http://127.0.0.1:7860` link in your browser.  

---

