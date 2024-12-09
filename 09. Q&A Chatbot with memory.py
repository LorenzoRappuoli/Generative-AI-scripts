from langchain_core.documents import Document
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_huggingface import HuggingFaceEmbeddings
import bs4
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import AIMessage,HumanMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder



### Model

load_dotenv()
groq_api_key=os.getenv("GROQ_API_KEY")
llm=ChatGroq(
    groq_api_key=groq_api_key,
    model="Llama3-8b-8192"
)

### Embedding model

os.environ['HF_TOKEN']=os.getenv("HF_TOKEN")
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


### Esempio

loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)

# carico tutti i documenti dal sito internet
docs=loader.load()

# creo i chunk di testo
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
splits = text_splitter.split_documents(docs)

# inizializzo il vectorstore come retriever

vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings
)
retriever = vectorstore.as_retriever()

### creo il mio promp template

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate(
    [
        ("system", system_prompt),
        ("human", "{input}")
    ]
)

### QA Chain (senza memoria)

# Create a chain for passing a list of Documents to a model.
QA_chain = create_stuff_documents_chain(llm, prompt)

# Create retrieval chain that retrieves documents and then passes them on.
rag_chain=create_retrieval_chain(retriever,QA_chain)

response = rag_chain.invoke(
    {"input":"What is Self-Reflection?"}
)

# print(response) # in output un dizionario



### Aggiungo la memoria


# contextualize_q_system_prompt è una stringa che definisce l'istruzione per il sistema, spiegando
# come riformulare una domanda di un utente in modo che sia comprensibile anche senza lo storico
# della conversazione. L'obiettivo è creare una domanda autonoma e chiara

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

# contextualize_q_prompt è un oggetto ChatPromptTemplate che definisce una struttura di prompt usando
# messaggi predefiniti. Contiene:
# Un messaggio di sistema con l'istruzione (contextualize_q_system_prompt).
# Un segnaposto per lo storico della conversazione (MessagesPlaceholder("chat_history")).
# Un messaggio umano che rappresenta l'input dell'utente (("{input}")).

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Questa funzione crea un retriever consapevole della cronologia delle conversazioni.
# Il retriever è responsabile di ricevere una query e restituire documenti pertinenti.
# In questo caso:
#
# Usa llm (un modello di linguaggio come GPT) e retriever (una struttura per il recupero dei documenti).
# Utilizza contextualize_q_prompt per riformulare le query considerando lo storico della chat, se presente.

# questo diventa il nostro "retriever"

history_aware_retriever=create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

# QA Chain

# system_prompt è un'istruzione per il modello in cui viene specificato come rispondere alle domande.
# Indica di utilizzare il contesto fornito per rispondere e di mantenere le risposte concise, con un massimo di tre frasi

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)


qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


question_answer_chain=create_stuff_documents_chain(llm,qa_prompt)

rag_chain=create_retrieval_chain(history_aware_retriever,question_answer_chain)

chat_history=[]
question="What is Self-Reflection"
response1=rag_chain.invoke({"input":question,"chat_history":chat_history})

chat_history.extend(
    [
        HumanMessage(content=question),
        AIMessage(content=response1["answer"])
    ]
)

question2="Can you describe it in italian?"
response2=rag_chain.invoke({"input":question2,"chat_history":chat_history})
print(response2['answer'])


### Altro modo per gestire la memoria

store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)


R1 = conversational_rag_chain.invoke(
    {"input": "What is Task Decomposition?"},
    config={
        "configurable": {"session_id": "abc123"}
    },  # constructs a key "abc123" in `store`.
)
print(R1['answer'])

R2 = conversational_rag_chain.invoke(
    {"input": "What are common ways of doing it?"},
    config={"configurable": {"session_id": "abc123"}},
)

print( R2['answer'])

print(f"store : {store}")