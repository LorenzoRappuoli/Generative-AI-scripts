
import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
import os
import time

# Setup iniziale e password
from dotenv import load_dotenv
load_dotenv()

os.environ['HF_TOKEN']=os.getenv("HF_TOKEN")
groq_api_key=os.getenv("GROQ_API_KEY")
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


## Streamlit
st.title("Conversational RAG With PDF uplaods and chat history")

## Seleziono il modello
engine=st.sidebar.selectbox("Select Open AI model",["gemma2-9b-it","Llama3-8b-8192","mixtral-8x7b-32768"])

## Seleziono i parametri per il modello
temperature=st.sidebar.slider("Temperature",min_value=0.0,max_value=1.0,value=0.5)
max_tokens = st.sidebar.slider("Max Tokens", min_value=10, max_value=1000, value=150)


llm = ChatGroq(
    groq_api_key=groq_api_key,
    model=engine,
    max_tokens=max_tokens,
    temperature=temperature
)

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only. 
    Please provide the most accurate response based on the question
    <context>
    {context}
    Question: {input}
    """
)

def create_vector_embedding():

    if "vectors" not in st.session_state:
        st.session_state.embeddings = embeddings
        st.session_state.loader = PyPDFLoader("temp.pdf") # Data ingestion
        st.session_state.docs = st.session_state.loader.load() # document loading
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        st.session_state.vectors = FAISS.from_documents(
            st.session_state.final_documents,
            st.session_state.embeddings
        )


prompt_user = st.text_input("Inserisci la tua domanda")

if st.button("Document Embedding"):
    create_vector_embedding()
    st.write("Il database vettoriale è pronto")

if prompt_user:

    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    start = time.process_time()
    response = retrieval_chain.invoke({"input":prompt_user})
    print(f"Response time: {time.process_time() - start}")

    st.write(response['answer'])

    with st.expander("Document similarity search"):
        for i,doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write('----------------------------')

