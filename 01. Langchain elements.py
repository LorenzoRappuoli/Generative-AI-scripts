### Libraries
import bs4
import os
from dotenv import load_dotenv
import json
import requests

from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import HTMLHeaderTextSplitter
from langchain_text_splitters import RecursiveJsonSplitter

from langchain_openai import OpenAIEmbeddings

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_chroma import Chroma as Chroma_lgc

######  Loading data

## 1. txt
loader_txt = TextLoader("./Dataset/01. Esempio txt.txt")
txt = loader_txt.load()

## 2. pdf
loader_pdf = PyPDFLoader("./Dataset/02. Esempio pdf.pdf")
pdf = loader_pdf.load()

## 3. web file
loader_web = WebBaseLoader(
    "https://embassies.gov.il/rome/AboutIsrael/State/Pages/Israel%20Defense%20Forces%20-IDF-.aspx",
    bs_kwargs=
    dict(parse_only = bs4.SoupStrainer(
        class_=("mfa_Layout_Content_Coll_2")))
                           )

web = loader_web.load()
#print(web)

## 4. wikipedia
wiki = WikipediaLoader(
    query="IDF",
    load_max_docs=2
).load()

## 5. Recursive

Recursive_loader = RecursiveUrlLoader(
    "https://www.leonardo.com/en/media-hub/",
     max_depth=3,
    # use_async=False,
    # extractor=None,
    # metadata_extractor=None,
    # exclude_dirs=(),
    # timeout=10,
    # check_response_status=True,
    # continue_on_failure=True,
    # prevent_outside=True,
    # base_url=None,
    # ...
).load()

#print(Recursive_loader)


#-------------------------------------------------------------------------------------------

###### Splitting

## 1. Recursive (generic text)

# This text splitter is the recommended one for generic text. It is parameterized by a list of characters.
# It tries to split on them in order until the chunks are small enough. The default list is ["\n\n", "\n", " ", ""].
# This has the effect of trying to keep all paragraphs (and then sentences, and then words) together as long as possible,
# as those would generically seem to be the strongest semantically related pieces of text.

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=50,
    chunk_overlap= 15
)

# non devo creare i documenti, sono già divisi
docs = text_splitter.split_documents(pdf)

#for d in docs:
#    print(d)

# file txt

with open('./Dataset/01. Esempio txt.txt') as file:
    testo = file.read()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=10
)

# in questo caso ho un testo txt, creo i documenti
txt = text_splitter.create_documents([testo])
#print(txt)

## 2. Recursive con separator

text_splitter2 = RecursiveCharacterTextSplitter(
    chunk_size=50,
    chunk_overlap= 15,
    separators="\n\n"
)

txt = text_splitter.create_documents([testo])
#print(txt)

## 3. HTML splitter

# HTMLHeaderTextSplitter is a "structure-aware" chunker that splits text at the HTML element level and adds metadata
# for each header "relevant" to any given chunk. It can return chunks element by element or combine elements with the same metadata,
# with the objectives of (a) keeping related text grouped (more or less) semantically and (b) preserving context-rich
# information encoded in document structures. It can be used with other text splitters as part of a chunking pipeline.

with open("./Dataset/esempio html.txt") as f: # direttamente come stringa
    testo_html = f.read()

headers_split = [
    ("h1", "Header 1"),
    ("h2", "Header 2"),
    ("h3", "Header 3")
]

html_splitter = HTMLHeaderTextSplitter(
    headers_to_split_on=headers_split
)

# print(html_splitter.split_text(testo_html))

## 4. url splitter (html)

url = "https://www.bbc.com/news/world-africa-68675452?at_campaign_type=owned&at_medium=emails&at_objective=awareness&at_ptr_type=email&at_ptr_name=salesforce&at_campaign=newsbriefing&at_email_send_date=20240404&at_send_id=4065327&at_link_title=https%3a%2f%2fwww.bbc.co.uk%2fnews%2fworld-africa-68675452&at_bbc_team=crm"

html_splitter = HTMLHeaderTextSplitter(
    headers_to_split_on = headers_split,
    return_each_element=True,
)

html_url_split = html_splitter.split_text_from_url(url)

#for h in html_url_split:
#    print(h)


## 4. json splitter
json_data = requests.get('https://api.smith.langchain.com/openapi.json').json()

json_splitter = RecursiveJsonSplitter(max_chunk_size=300)
json_chunks = json_splitter.split_json(json_data)
#print(json_chunks)

# json diviso in documents
#docs = json_splitter.create_documents(texts=json_data)


#-------------------------------------------------------------------------------------------

###### Embedding (FROM TEXT TO VECTORS)

## 1. OPENAI

# carico le variabili ambientali
load_dotenv()

#OpenAI Keys
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

# test caricamento chiavi API
#openai_api_key = os.getenv("OPENAI_API_KEY")
#
#if openai_api_key is None:
#    print("La chiave OPENAI_API_KEY non è stata trovata")
#else:
#    print(f"Chiave API caricata: {openai_api_key[:5]}...")  # Stampa solo i primi 5 caratteri per essere sicuro di caricare il file .env


#embeddings = OpenAIEmbeddings(
#    model="text-embedding-3-large",
#    dimensions = 1024,
#    # With the `text-embedding-3` class
#    # of models, you can specify the size
#    # of the embeddings you want returned.
#    # dimensions=1024
#)
#
loader_txt2 = TextLoader("./Dataset/01. Esempio txt.txt")
txt2 = loader_txt.load()
#
#
final_documents = text_splitter.split_documents(txt2)
##print(final_documents)
#
#
## vector embedding and store
#
#db = Chroma.from_documents(
#    final_documents, # base dati
#    embeddings, # modello
#)
#
### Esempio
#
#query = "I due accordi sono stati firmati alla Casa Bianca sotto l'auspicio del presidente"
#
#retrived_result = db.similarity_search(query = query)
#
#print(retrived_result)


## Ollama embedding

embeddings_ollama = (
    OllamaEmbeddings(
        model = "nomic-embed-text"
    )# by default llama2
)

result1 = embeddings_ollama.embed_documents(final_documents)

print('testo con Ollama [nomic-embed-text] model: ')
for r in result1:
    print(r)

#print(result1)

result2 = embeddings_ollama.embed_query(["Ciao, come stai?"])
print('--------------------------------------')
print('frase singola: ')
print(result2)



### hugging faces

print('-----------------huggin face-----------------')

os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")

embeddings_hf = HuggingFaceEmbeddings(
    model_name  = "all-MiniLM-L6-v2"
)

text = 'Esempio da embeddare'

query_result= embeddings_hf.embed_query(text)
print(query_result)



#-------------------------------------------------------------------------------------------

##### Vectorstores

print('------------Vectorstores-----------------')
###  1. Faiss

# uso Ollama
db = FAISS.from_documents(
    final_documents,
    embeddings_ollama
)

query = "Cosa a proposito della guerra?"

doc_answer = db.similarity_search(query)

print(doc_answer[0].page_content)

# Retriver

retriver = db.as_retriever()
documento = retriver.invoke(query)
print(documento[0])

# Similiarity with score

doc_answer_score = db.similarity_search_with_score(query)
print(doc_answer_score[0])


## è possibile salvarlo in Locale
# db.save_local('faiss_db')
# db.save_local('faiss_db') # salvo in locale
# new_db = FAISS.load_local(
#     "faiss_db",
#     embeddings_ollama,
#     allow_dangerous_deserialization=True # serve per caricare da locale
# )


###  1. Faiss

vectordb = Chroma_lgc.from_documents(
    final_documents,
    embeddings_ollama
)

query = "Cosa a proposito della guerra?"

doc_answer = vectordb.similarity_search(query)

print(doc_answer[0].page_content)

## salvo in locale il db vettoriale

# vectordb = Chroma_lgc.from_documents(
#     final_documents,
#     embeddings_ollama,
#     persist_directory= "./chroma_db"
# )


# retriver con Chroma

retriver = vectordb.as_retriever()
print(retriver.invoke(query)[0].page_content)
