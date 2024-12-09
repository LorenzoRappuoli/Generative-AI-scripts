import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

### groq api key
groq_api_key=os.getenv("GROQ_API_KEY")

### Model
model=ChatGroq(
    model="Gemma2-9b-It",
    groq_api_key=groq_api_key
)

# Creo le tipologia di messaggio da passare al modello
messages = [
    SystemMessage(content="Translate the following from English to Spanish, don't add any comment"),
    HumanMessage(content="Hello How are you?")
]

result = model.invoke(messages)

## Parser del content
parser = StrOutputParser()

#print(parser.invoke(result))

## Lcel - concateno i componenti
chain = model|parser
print(chain.invoke(messages))

### prompt Template

generic_template = "Translate the following into {language} without adding any comment"

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", generic_template),
        ("user", "{text}")
     ]
)

esempio_prompt = prompt.invoke(
    {
        "language":"Latin",
        "text":"Hello, how are you?"
     }
)
print(esempio_prompt.to_messages())

chain_lcel = prompt|model|parser
result_lcel = chain_lcel.invoke(
    {
        "language": "Latin",
        "text": "Hello, how are you?"
    }
)

print(result_lcel)