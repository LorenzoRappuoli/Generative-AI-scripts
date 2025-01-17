import streamlit as st
from pathlib import Path
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from sqlalchemy import create_engine
import sqlite3
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

st.set_page_config(page_title="LangChain: Chat with SQL DB", page_icon="🦜")
st.title("🦜 LangChain: Chat with SQL DB")

# Carica le variabili d'ambiente dal file .env
load_dotenv()

# llm
groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(
    groq_api_key=groq_api_key,
    #model="Llama3-8b-8192",
    model = "llama3-70b-8192",
    max_tokens=1024,
    temperature=0.5
)

# db MSSQL SEVER in locale
#conn_str = "mssql+pyodbc:///?odbc_connect=DRIVER={SQL Server};SERVER=BIPITA-PF48NBM5\SQLEXPRESS;DATABASE=Langchain agent;Trusted_Connection=yes;CHARSET=utf8;"
conn_str = "mssql+pyodbc:///?odbc_connect=DRIVER={ODBC Driver 17 for SQL Server};SERVER=BIPITA-PF48NBM5\SQLEXPRESS;DATABASE=Langchain agent;Trusted_Connection=yes;"

db = SQLDatabase.from_uri(conn_str)
db.get_usable_table_names()

# Creo il toolkit SQL
toolkit=SQLDatabaseToolkit(db=db,llm=llm)

# Creo l'agente
agent=create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
)

# Inizializzo o resetto la session state
if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_query=st.chat_input(placeholder="Ask anything from the database")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        streamlit_callback=StreamlitCallbackHandler(st.container())
        response=agent.run(user_query,callbacks=[streamlit_callback])
        st.session_state.messages.append({"role":"assistant","content":response})
        st.write(response)