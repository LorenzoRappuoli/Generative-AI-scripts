import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

###### OPENAI

### Keys

os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

## Langsmith
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

llm = ChatOpenAI(
    # api_key = non serve perchè ho caricato le variabili ambientali
    model = "gpt-4o"
)

### Domanda iniziale

# result = llm.invoke("Who is Dante Alighieri?")
# print(result)

### uso il ChatPromptTemplate per customizzare il mio llm

prompt = ChatPromptTemplate.from_messages(
    [
        ("system" , "You are an expert historian. Provide me answers based on the question"), # customizzo l'agente
        ("user", "{input}")
    ]
)

## creo la catena: prompt da seguire + llm da usare

#chain = prompt|llm
#
#response = chain.invoke({"input":"Can you tell me who was Alexander the Great?"})
#print(response)

## per miglirorare la forma dell'output

output_parser=StrOutputParser()
chain=prompt|llm|output_parser

response=chain.invoke({"input":"Can you tell me about Francesco Petrarca?"})
print(response)


