from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from langserve import add_routes

load_dotenv()

### groq api key
groq_api_key=os.getenv("GROQ_API_KEY")

### Model
model=ChatGroq(
    model="Gemma2-9b-It",
    groq_api_key=groq_api_key
)

### Prompt + Parser

generic_template = "Translate the following into {language} without adding any comment"

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", generic_template),
        ("user", "{text}")
     ]
)

parser = StrOutputParser()

### chain

chain_groq = prompt|model|parser

### definizione APP

app=FastAPI(title="Langchain Server",
            version="1.0",
            description="A simple API server using Langchain runnable interfaces")

add_routes(
    app,
    chain_groq,
    path="/chain"
)

if __name__=="__main__":
    import uvicorn
    uvicorn.run(app,host="127.0.0.1",port=8000)