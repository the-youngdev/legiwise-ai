# ⚖️ LegiWise AI: Project Status & Roadmap 
# ⚖️ LegiWise AI  

![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![HuggingFace](https://img.shields.io/badge/🤗-Transformers-orange)
![Gradio](https://img.shields.io/badge/Gradio-Interface-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Status](https://img.shields.io/badge/Status-Active-success)


**Date:** September 7, 2025  
**Status:** Core Features Implemented · Final Debugging Phase  

---

## 1. Project Overview  
The goal of the **LegiWise AI** project is to create a comprehensive, AI-powered legal assistant.  
All major features have been built, and the application is in the final phase of **debugging and stabilization**.  

---

## 2. What We Have Built (Confirmed Working)  

- 📑 **Document Loading & OCR**: Successfully loads **PDF, DOCX, and image files**. OCR engine correctly extracts text from scanned documents.  
- 🧭 **Legal Navigator**: Pre-loaded legal guides and dedicated Q&A fully functional.  
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

