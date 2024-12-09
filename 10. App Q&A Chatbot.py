import streamlit as st
import openai
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import os

import os
from dotenv import load_dotenv
load_dotenv()

## Langchain & Langsmith Tracking
#os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")

groq_api_key=os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]="Simple Q&A Chatbot With OPENAI"

## Prompt Template
prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant. Please answer to the user queries in italian"),
        ("user","Question:{question}")
    ]
)


def generate_response(question,engine, temperature, max_tokens):
    llm=ChatGroq(
        groq_api_key=groq_api_key,
        model=engine,
        max_tokens = max_tokens,
        temperature = temperature
    )
    output_parser=StrOutputParser()
    chain=prompt|llm|output_parser
    answer=chain.invoke({'question':question})
    return answer

## Titolo dell'app
st.title("Q&A Chatbot With OpenAI")

## Settings
st.sidebar.title("Settings")

## Seleziono il modello
engine=st.sidebar.selectbox("Select Open AI model",["gemma2-9b-it","Llama3-8b-8192","mixtral-8x7b-32768"])

## Seleziono i parametri per il modello
temperature=st.sidebar.slider("Temperature",min_value=0.0,max_value=1.0,value=0.5)
max_tokens = st.sidebar.slider("Max Tokens", min_value=10, max_value=1000, value=150)

## Corpo centrale
st.write("Scrivi qui la tua domanda")
user_input=st.text_input("You:")

if user_input:
    response=generate_response(user_input,engine,temperature,max_tokens)
    st.write(response)

elif user_input:
    st.warning("Some issue is occurring")
else:
    st.write("Please provide the user input")