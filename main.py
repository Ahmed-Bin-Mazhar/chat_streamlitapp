import streamlit as st
import PyPDF2

def read_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

st.title("PDF Reader App")

st.write("Upload a PDF file to read its content.")

try:
   uploaded_file = st.file_uploader("Choose a file", type=["pdf"])
   
   if uploaded_file:
      pdf_text = read_pdf(uploaded_file)
      st.write(pdf_text)
    




except Exception as ex:
    print("[+] Exception :{str(ex)}")