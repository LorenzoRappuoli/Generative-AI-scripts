import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain.chains import create_retrieval_chain

load_dotenv()


###### OPENAI

### Keys

os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

## Langsmith
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")


##### Scraping il sito internet
link = "https://www.ansa.it/sito/notizie/cronaca/2024/10/23/intercettati-missili-di-hezbollah-lanciati-su-tel-aviv_dc8e074a-e3f1-4c84-b47f-878f1220e402.html"
loader = WebBaseLoader(link)
docs = loader.load()

##### Divido in chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=20
)
documents = text_splitter.split_documents(docs)
#print(documents)

##### Embeddings + vectorstore db
embeddings = OpenAIEmbeddings()

vectorstoredb = FAISS.from_documents(
    documents,
    embeddings
)

##### query nel vector db

query = "Cosa ha fatto Hezbollah"

result = vectorstoredb.similarity_search(query)
#print(result[0].page_content)

##### retrival chain, document chain

## llm model
llm = ChatOpenAI(
    # api_key = non serve perchè ho caricato le variabili ambientali
    model = "gpt-4o"
)

## document chain

prompt = ChatPromptTemplate.from_template(
    """
    Answer the following question based only on the provided context : 
    <context>
    {context}
    </context>
    """
)
## https://python.langchain.com/v0.1/docs/modules/chains/
document_chain = create_stuff_documents_chain(
    llm = llm,
    prompt= prompt
)

#document_chain.invoke(
#    {
#        "input":query,
#        "context": [Document(page_content=)] # qui passerei i chunk come lista di document
#    }
#)

## Input--->Retriever--->vectorstoredb
## Create retrieval chain that retrieves documents and then passes them on.
retriever=vectorstoredb.as_retriever()
retrieval_chain=create_retrieval_chain(
    retriever, # da dove viene l'info
    document_chain # responsabile per il context nel prompt
)

## Response from LLM
response = retrieval_chain.invoke(
    {
        "input":query
    }
)
print(response['answer'])
print('----')
print(response['context'])


