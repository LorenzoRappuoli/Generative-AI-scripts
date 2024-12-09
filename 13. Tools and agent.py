import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun,DuckDuckGoSearchRun
from langchain.callbacks import StreamlitCallbackHandler
from langchain.agents import AgentExecutor, create_react_agent
from dotenv import load_dotenv
import os
from langchain import hub
# Carica le variabili d'ambiente dal file .env
load_dotenv()

## Arxiv e wikipedia Tools
arxiv_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv=ArxivQueryRun(api_wrapper=arxiv_wrapper)

api_wrapper=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=200)
wiki=WikipediaQueryRun(api_wrapper=api_wrapper)

# search web
search=DuckDuckGoSearchRun(name="Search")


st.title("🔎 LangChain - Chat with search")

# LangChain Streamlit Agent examples at [github.com/langchain-ai/streamlit-agent](https://github.com/langchain-ai/streamlit-agent).

##
groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model="Llama3-8b-8192",
    max_tokens=1024,
    temperature=0.5
)

## Creo le variabili per la sessione


if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {"role":"assisstant","content":"Hi,I'm a chatbot who can search the web. How can I help you?"}
    ]

# tramite chat_message creo il dialogo
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

# messaggio da inviare
if prompt:=st.chat_input(placeholder="What is machine learning?"):
    st.session_state.messages.append({"role":"user","content":prompt})
    st.chat_message("user").write(prompt)

    prompt_agent = hub.pull("hwchase17/react") # serve per far ragionare l'agente
    tools=[search,arxiv,wiki]
    search_agent = create_react_agent(llm, tools, prompt_agent)
    agent_executor = AgentExecutor(agent=search_agent, tools=tools)

    with st.chat_message("assistant"):
        response=agent_executor.invoke({"input": prompt})
        st.session_state.messages.append({'role':'assistant',"content":response})
        st.write(response['output'])
















