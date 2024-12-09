# Importa le librerie necessarie
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
import os
from dotenv import load_dotenv

# Carica le variabili d'ambiente dal file .env
load_dotenv()

# Funzione per ottenere la cronologia della chat dalla sessione
def get_session_history(session: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

# Imposta le chiavi API dalle variabili d'ambiente
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")
groq_api_key = os.getenv("GROQ_API_KEY")

# Impostazione degli embeddings con il modello HuggingFace
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Configurazione dell'interfaccia Streamlit
st.title("Conversational RAG With PDF uploads and chat history")
engine = st.sidebar.selectbox("Select Open AI model", ["gemma2-9b-it", "Llama3-8b-8192", "mixtral-8x7b-32768"])
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.5)
max_tokens = st.sidebar.slider("Max Tokens", min_value=10, max_value=1000, value=150)

uploaded_files = st.file_uploader("Choose A PDF file", type="pdf", accept_multiple_files=True)


# Crea un'istanza del modello ChatGroq
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model=engine,
    max_tokens=max_tokens,
    temperature=temperature
)

# Gestione della sessione e caricamento file
session_id = st.text_input("Session ID", value="default_session")
if 'store' not in st.session_state:
    st.session_state.store = {}


# Carica i file PDF caricati
if uploaded_files:
    documents = []
    for uploaded_file in uploaded_files:
        temppdf = f"./temp.pdf"
        with open(temppdf, "wb") as file:
            file.write(uploaded_file.getvalue())

        loader = PyPDFLoader(temppdf)
        docs = loader.load()
        documents.extend(docs)

    # Inizializza il vettoriale dei documenti
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(documents)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

    # Crea un retrieve che utilizza gli embeddings vettoriali
    retriever = vectorstore.as_retriever()

    # Questo blocco definisce un prompt che aiuta a riformulare una domanda dell'utente in modo che possa essere
    # considerata indipendente dal contesto, utilizzando la cronologia della chat.

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question"
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    # Crea una catena di recupero storico consapevole
    # Crea un oggetto che prende in considerazione la cronologia della chat quando recupera
    # documenti pertinenti per rispondere a una domanda.
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)


    # Fornisce istruzioni al modello sul come comportarsi, cioè utilizzare il contesto recuperato per
    # rispondere alle domande in modo preciso e conciso.

    # Definisce il prompt del sistema per la risposta alle domande
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise.\n\n{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    # Crea una catena di documenti di risposta a domande
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # Crea una catena che esegue la retrieval e passa i documenti
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

# Gestione dell'input utente e della risposta
user_input = st.text_input("Your question:")
if user_input:
    session_history = get_session_history(session_id)
    response = conversational_rag_chain.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": session_id}}
    )
    st.write(st.session_state.store)
    st.write("Assistant:", response['answer'])
    st.write("Chat History:", session_history.messages)