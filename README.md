# 📘 PDF Question Answering with Streamlit, LangChain & Groq

This project lets you **chat with a PDF** using [Streamlit](https://streamlit.io/), [LangChain](https://python.langchain.com/), [Groq LLMs](https://groq.com/), and vector embeddings with [FAISS](https://github.com/facebookresearch/faiss).  
You can upload any PDF file, and the app will let you ask questions based on its contents.

---

## 🚀 Features
- Upload and read PDFs (`PyPDF2` based reader).  
- Chunk & embed text with **HuggingFace embeddings**.  
- Store and query embeddings with **FAISS vector store**.  
- Use **Groq LLM** for question answering.  
- Interactive **chat interface** powered by **Streamlit**.  
- Maintains chat history for contextual Q&A.  

---

## 📂 Project Structure
.
├── app.py # Main Streamlit application
├── .env # Environment variables
└── requirements.txt

## 📂 Venv
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows

## Requirement.txt installation 
pip install -r requirements.txt



## Environment Variables
EMBED_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
GROQ_API_KEY=**your_groq_api_key_here**
GROQ_LLM_MODEL=llama2-70b-4096   # Or any supported Groq model


## streamlit run app.py
streamlit run app.py
