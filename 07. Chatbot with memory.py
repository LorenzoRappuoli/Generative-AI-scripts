from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq

from langchain_community.chat_message_histories import ChatMessageHistory
# Questa classe è specifica per la gestione della cronologia dei messaggi nelle implementazioni della community di LangChain
# viene utilizzata per storicizzare messaggi in applicazioni sviluppate o estese dalla community, sfruttando meccanismi di archiviazione alternativi o integrati con soluzioni specifiche di backend.
from langchain_core.chat_history import BaseChatMessageHistory
# Rappresenta la classe base o astratta per gestire la cronologia dei messaggi in LangChain.
# È generalmente estesa da altre classi per definire la logica di archiviazione, come storicizzare, recuperare e gestire i messaggi in una conversazione.
from langchain_core.runnables.history import RunnableWithMessageHistory
# Questa classe viene utilizzata quando si vuole che un oggetto o una funzione eseguibile mantenga il contesto di una conversazione, permettendo di
# accedere alla cronologia anche mentre si eseguono task paralleli o moduli iterativi.
import os
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder

from langchain_core.messages import SystemMessage,trim_messages
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough



### groq api key
load_dotenv()
groq_api_key=os.getenv("GROQ_API_KEY")

### Model
model=ChatGroq(
    model="Gemma2-9b-It",
    groq_api_key=groq_api_key
)

### Conversazione con memoria aggiunta manualmente
result = model.invoke(
    [
        HumanMessage(content="Hi, my name is Lorenzo and I'm a data consultant"),
        AIMessage(content="Hi Lorenzo, do you like your job?"),
        HumanMessage(content="Not really, I would prefer to be a photographer")
     ]
)

#print(result.content)

### creo storage per salvare i session id delle conversazioni per recuperare memoria e modello

store = {} # memorizza le varie session id

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """

    :param session_id: indica che il parametro session_id deve essere di tipo str (stringa)
    :return: -> BaseChatMessageHistory specifica che il valore restituito dalla funzione sarà di tipo BaseChatMessageHistory
    """
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# aggiungo la componente memoria
with_message_history = RunnableWithMessageHistory(
    model, # llm
    get_session_history # memoria della chat
)
# https://api.python.langchain.com/en/latest/runnables/langchain_core.runnables.history.RunnableWithMessageHistory.html

# nelle configurazione fisso un id che mi permette di identificare la conversazione
config = {
    "configurable":{"session_id":"chat1"}
}

# due frasi che entrano nella conversazione

response = with_message_history.invoke(
    [HumanMessage(content="Hi, my name is Lorenzo and I'm a data consultant")],
    config = config
)

response2 = with_message_history.invoke(
    [HumanMessage(content="What is my name and my job?")],
    config = config
)
#print(response2.content)
#print(with_message_history.get_session_history("chat1"))



### Chatbot utilizzando un Prompt template

#1. System Message: This is used to baseline the AI assistant. If you want to set some behaviour of your assistant. This is the place to do it.
#2. HumanMessage: All the user inputs are to stored in this
#3. AIMessage: This stores the response from the LLM.

prompt2=ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful and generous assistant.You want to help other people,but you don't want to get screwed. Answer in italian like an Italian from the Sicily with a mafia approach. "),
        MessagesPlaceholder(variable_name="messages")
    ]
)


chain2 = prompt2|model

result1 = chain2.invoke(
    {
        # messages è la chiave
        "messages":[HumanMessage(
            content="Hi my name is Lorenzo, I need your help"
        )]
    }
)

#print(result1.content)

# aggiungo la memoria

with_message_history2 = RunnableWithMessageHistory(
    chain2,
    get_session_history=get_session_history
)


# aggiungo la memoria

config3 = {"configurable": {"session_id": "chat3"}}

response3 = with_message_history.invoke(
    [HumanMessage(content="Hi My name is Lorenzo")],
    config=config3
)

response4 = with_message_history.invoke(
    [HumanMessage(content="What's my name?")],
    config=config3,
)

#print(response4.content)

## Componente dinamica


prompt5 = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability in {language}.",
        ),
        MessagesPlaceholder(variable_name="messages"), # qui memorizzo la variabile
    ]
)
chain5 = prompt5|model
response5=chain5.invoke({"messages":[HumanMessage(content="Hi My name is Lorenzo")],"language":"Italian"})
print(response5.content)

### Uniamo la memoria e la componente dinamica

config4 = {"configurable": {"session_id": "chat4"}}

# aggiungo la baseline
# Specifica input_messages_key="messages" per indicare che i messaggi da registrare nella cronologia si trovano nella chiave "messages" dell'input.
prompt6 = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability in {language} as a {job}.",
        ),
        MessagesPlaceholder(variable_name="messages")
    ]
)

# creo la catena
chain5 = prompt6|model

with_message_history5=RunnableWithMessageHistory(
    chain5,
    get_session_history,
    input_messages_key="messages" # segnaposto (MessagesPlaceholder) che verrà riempito con i messaggi forniti durante l'esecuzione.
)

repsonseA =with_message_history5.invoke(
    {'messages': [HumanMessage(content="Hi,I am Lorenzo. I'm 31 years old")],"language":"Italian", "job":"Military officer"},
    config=config4
)

repsonseB =with_message_history5.invoke(
    {'messages': [HumanMessage(content="How old am I?")],"language":"Italian", "job":"Military officer"},
    config=config4
)

# print(repsonseB.content)


### Gestire la conversazione
# Per evitare che la conversazione cresca senza limiti e il numero di dati superi la capacità di gestione -> trim message


trimmer=trim_messages(
    max_tokens=100,
    strategy="last", # si focalizza sugli ultimi messaggi
    token_counter=model, # modello che conta i token
    include_system=True, # includi i messaggi di sistema
    allow_partial=False, # include solo messaggi inter
    start_on="human" # parti dal primo messaggio di Human
    # end_on=("human", "ai") # con cosa finisce la conversazione
)


messages = [
    SystemMessage(content="you're a good assistant"),
    HumanMessage(content="hi! I'm bob"),
    AIMessage(content="hi!"),
    HumanMessage(content="I like vanilla ice cream"),
    AIMessage(content="nice"),
    HumanMessage(content="whats 2 + 2"),
    AIMessage(content="4"),
    HumanMessage(content="thanks"),
    AIMessage(content="no problem!"),
    HumanMessage(content="having fun?"),
    AIMessage(content="yes!"),
]

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability in {language}.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

chain7 = (
    RunnablePassthrough.assign(
        messages = itemgetter("messages")|trimmer
    )| prompt
    | model
)

response7 = chain7.invoke(
    {
        "messages": messages + [HumanMessage(
            content="What is my favourite ice cream flavour?"
        )],
        "language" : "Italian"
    }
)

# print(response7.content)
# inseriamo la memoria

with_message_history6 = RunnableWithMessageHistory(
    chain7,
    get_session_history,
    input_messages_key="messages"
)

config = {"configurable":{"session_id":"chat7"}}

response6 = with_message_history6.invoke(
    {
        "messages": messages + [HumanMessage(content="whats did I ask?")],
        "language": "English",
    },
    config=config,
)

print(response6.content)

response7 = with_message_history6.invoke(
    {
        "messages": messages + [HumanMessage(content="and before that?")],
        "language": "English",
    },
    config=config,
)
print(response7)
print(response7.content)
print(response7.usage_metadata['input_tokens'])
print(response7.usage_metadata['output_tokens'])