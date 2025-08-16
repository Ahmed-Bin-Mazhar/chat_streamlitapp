import os
import PyPDF2
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()
# loading variables from environment file
EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL_NAME")
os.environ["GROQ_API_KEY"] = os.environ.get("GROQ_API_KEY")
# Initializing Embedding Model
embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)

print(' [+] Embedding Model Loaded')

# Initializing LLM
llm = ChatGroq(
    temperature=0,
    model_name=os.environ.get("GROQ_LLM_MOODEL"),
)

print(' [+] LLM Model Loaded')


# Initializing Text Splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=20,
    chunk_overlap=20,
    length_function=len,
)

print(' [+] Text Splitter Initialized')


# Defining Function to read Pdf file
def read_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for pdf_page in range(len(reader.pages)):
        page = reader.pages[pdf_page]
        text += page.extract_text()
    return text

try:
   uploaded_file = st.file_uploader("Choose a file", type=["pdf"])
   
   if uploaded_file:
      pdf_text = read_pdf(uploaded_file)
      
      text=RecursiveCharacterTextSplitter.split_text(
          text=pdf_text,
          text_splitter=text_splitter,
          
        )
      st.write("Text Splitter Output:" + str(text))


except Exception as ex:
    print("[+] Exception :",{str(ex)})