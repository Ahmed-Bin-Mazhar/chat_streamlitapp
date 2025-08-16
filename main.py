import os
import PyPDF2
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter


load_dotenv()

# loading variables from environment file
EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL_NAME")
os.environ["GROQ_API_KEY"] = os.environ.get("GROQ_API_KEY")

# Initializing Embedding Model
embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)

print("[+] Embedding Model Loaded")

# Initializing LLM
llm = ChatGroq(
    temperature=0,
    model_name=str(os.environ.get("GROQ_LLM_MODEL")),
)

print("[+] LLM Model Loaded")

# Text Splitter
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=500,
    chunk_overlap=50,
)


print("[+] Text Splitter Loaded")


# Defining Function to read Pdf file
def read_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for pdf_page in range(len(reader.pages)):
        page = reader.pages[pdf_page]
        text += page.extract_text()
    return text


try:
    uploaded_file = st.file_uploader("Choose a file", type="pdf")

    if uploaded_file:
        pdf_text = read_pdf(uploaded_file)

        splitted_text = text_splitter.split_text(pdf_text)
        print("[+] Splitting Done ")

        document = FAISS.from_texts(splitted_text, embedding=embedding_model)
        qa = load_qa_chain(llm=llm, chain_type="stuff")

        if user_prompt := st.chat_input("Say something"):

            st.write(f"User : {user_prompt}")
            retriver = document.similarity_search(user_prompt)

            print("[+] Document Retrive")

            llm_response = qa.invoke(
                {"input_documents": retriver, "question": user_prompt}
            )
            st.write(f"Asistant Response : {llm_response["output_text"]}")

except Exception as ex:
    print(f"[+] Exception:{str(ex)}")
