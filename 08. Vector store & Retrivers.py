from langchain_core.documents import Document
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.runnables import RunnableLambda
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough



# LangChain implements a Document abstraction, which is intended to represent a unit of text and associated metadata.
# It has two attributes:
#   - page_content: a string representing the content;
#   - metadata: a dict containing arbitrary metadata.

documents = [
    Document(
        page_content="Dogs are great companions, known for their loyalty and friendliness.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Cats are independent pets that often enjoy their own space.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Goldfish are popular pets for beginners, requiring relatively simple care.",
        metadata={"source": "fish-pets-doc"},
    ),
    Document(
        page_content="Parrots are intelligent birds capable of mimicking human speech.",
        metadata={"source": "bird-pets-doc"},
    ),
    Document(
        page_content="Rabbits are social animals that need plenty of space to hop around.",
        metadata={"source": "mammal-pets-doc"},
    ),
]

### Model

load_dotenv()
groq_api_key=os.getenv("GROQ_API_KEY")
#os.environ["HF_TOKEN"]=os.getenv("HF_TOKEN")
llm=ChatGroq(
    groq_api_key=groq_api_key,
    model="Llama3-8b-8192")


### Embedding Model

embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


### Vectorstore Chroma

vectorstore=Chroma.from_documents(
    documents,
    embedding=embeddings)

#ricerca = vectorstore.similarity_search("cat")

# Run similarity search with distance
# The default distance is l2
ricerca = vectorstore.similarity_search_with_score("cat")

#for r in ricerca:
#    print(r)

# esempio risultato
# Document(metadata={'source': 'mammal-pets-doc'}, page_content='Cats are independent pets that often enjoy their own space.'), 0.9351057410240173)


### Retrivers

# LangChain VectorStore objects do not subclass Runnable, and so cannot immediately be integrated into LangChain
# Expression Language chains.
# LangChain Retrievers are Runnables, so they implement a standard set of methods (e.g., synchronous and asynchronous
# invoke and batch operations) and are designed to be incorporated in LCEL chains.

# RunnableLambda converts a python callable into a Runnable.
# Wrapping a callable in a RunnableLambda makes the callable usable within either a sync or async context.
# RunnableLambda can be composed as any other Runnable and provides seamless integration with LangChain tracing.

# Il vectorstore non è un elemento di Langchain che può essere integrato in una catena
# con Runnable lo diventa
retriever=RunnableLambda(vectorstore.similarity_search).bind(k=1) # top result
ricerca2 = retriever.batch(["cat","dog"]) # giro due input (parametri) che verranno eseguiti separatamente

#for r in ricerca2:
    #print(r)


## si può far funzionare il vectorstore come un retriver con "as_retriver"
## questo è il processo consigliato

retriever=vectorstore.as_retriever(
    search_type="similarity", # metodologia di punteggio per i risultati
    search_kwargs={"k":1} # top search
)
ricerca3 = retriever.batch(["cat","dog"])

##for r in ricerca3:
  ##  print(r)


### RAG con Vectorstore


# RunnablePassthrough on its own allows you to pass inputs unchanged. This typically is used in
# conjuction with RunnableParallel to pass data through to a new key in the map.

message = """
Answer this question using the provided context only.

{question}

Context:
{context}
"""
prompt = ChatPromptTemplate.from_messages([("human", message)])

rag_chain={"context":retriever,"question":RunnablePassthrough()}|prompt|llm

#  Questo dizionario rappresenta un oggetto in cui:
# retriever è una funzione o un oggetto che fornisce il contesto necessario alla risposta, probabilmente
# cercando nei documenti o in una base di dati.
# RunnablePassthrough() è un oggetto che semplicemente passa la domanda così com'è. -> è come invoke per il vectorstore as retriver

response=rag_chain.invoke("tell me about dogs")
print(response.content)
