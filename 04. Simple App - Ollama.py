import os
from dotenv import load_dotenv
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st
from langchain_core.output_parsers import StrOutputParser

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

## Promp template
## The prompt to chat models/ is a list of chat messages.
## Each chat message is associated with content, and an additional parameter called role.
## For example, in the OpenAI Chat Completions API, a chat message can be associated with an AI
## assistant, a human or a system role.

prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant. Please respond to the question asked"),
        ("user","Question:{question}")
    ]
)

## streamlit

st.title("Langchain Demo Llama3")
input = st.text_input("What question do you have in mind?")

## OLLAMA

llm = Ollama(model="llama3")
output_parser = StrOutputParser()
chain = prompt|llm|output_parser

if input:
    response = chain.invoke(
        {"question":input}
    )

    st.write(
        response
    )