import os
from dotenv import load_dotenv
from langchain.chains.summarize import load_summarize_chain
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st
import tempfile

# Carica variabili d'ambiente dal file .env
load_dotenv()

# Configurazione Streamlit
st.set_page_config(page_title="Riassunto notizie da PDF", page_icon="🦜")
st.title("🦜 Riassunto notizie da PDF")

# Carico PDF
uploaded_file = st.file_uploader('Choose your .pdf file', type="pdf")

# Chiave API Groq
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.error("Errore: la chiave API Groq non è configurata correttamente.")

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model="llama3-70b-8192",
    max_tokens=1024,
    temperature=0.5
)

if uploaded_file:
    if st.button("Summarize the Content"):
        with st.spinner("Waiting..."):
            try:
                # Salva il file PDF temporaneamente
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                    temp_pdf.write(uploaded_file.read())
                    temp_pdf_path = temp_pdf.name

                # Carica il documento dal PDF
                loader = PyPDFLoader(temp_pdf_path)
                docs = loader.load_and_split()

                # Dividi il testo in chunks
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                final_documents = text_splitter.split_documents(docs)

                # Esegui la catena di riepilogo
                chain = load_summarize_chain(
                    llm=llm,
                    chain_type="refine",
                    verbose=False
                )
                output_summary = chain.invoke(final_documents)
                st.success("Riassunto completato!")
                st.write(output_summary["output_text"])

            except Exception as e:
                st.error(f"Errore durante l'elaborazione: {e}")

            finally:
                # Rimuovi il file temporaneo
                if os.path.exists(temp_pdf_path):
                    os.remove(temp_pdf_path)
