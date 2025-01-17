import streamlit as st
from langchain.document_loaders import TextLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.vectorstores import FAISS
from langchain.retrievers import ContextualCompressionRetriever
#from langchain.retrievers.document_compressors import CohereRerank
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from dotenv import load_dotenv
import os

# Carica variabili d'ambiente dal file .env
load_dotenv()

os.environ['HF_TOKEN']=os.getenv("HF_TOKEN")
groq_api_key=os.getenv("GROQ_API_KEY")

# Configurazione Streamlit
st.set_page_config(page_title="Ricerca ibrida", page_icon="🦜")
st.title("🦜 Ricerca ibrida")

# Funzione per ottenere la cronologia della chat dalla sessione
def get_session_history(session: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

# Modelli
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model="llama3-70b-8192",
    max_tokens=1024,
    temperature=0.5
)

# Caricamento del file PDF
pdf_file = "Risultati.pdf"
loader = PyPDFLoader(pdf_file)

# Caricamento dei documenti
documents = loader.load()

# Suddivisione dei documenti in chunk
text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
text_splits = text_splitter.split_documents(documents)

# Vectorstore

vectorstore = FAISS.from_documents(text_splits, embeddings)

# retriver vectors + matrice sparsa

retriever_vectordb = vectorstore.as_retriever(search_kwargs={"k": 5})
keyword_retriever = BM25Retriever.from_documents(text_splits)
keyword_retriever.k = 5
ensemble_retriever = EnsembleRetriever(retrievers=[retriever_vectordb,keyword_retriever],
                                       weights=[0.5, 0.5])

# Manca la parte di rerank, da aggiungere facendo un account su Cohere

# Template per il llm

template = """
<|system|>>
You are an AI Assistant that follows instructions extremely well.
Please be truthful and give direct answers with a long explanation in italian. 
Please tell 'I don't know' if user query is not in CONTEXT

CONTEXT: {context}
</s>
<|user|>
{query}
</s>
<|assistant|>
"""

prompt = ChatPromptTemplate.from_template(template)
output_parser = StrOutputParser()

chain = (
    {"context": ensemble_retriever, "query": RunnablePassthrough()}
    | prompt
    | llm
    | output_parser
)

prompt_user = st.text_input("Inserisci la tua domanda")

if prompt_user:
    response = chain.invoke(prompt_user)
    st.write(response)
