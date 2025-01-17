import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader,UnstructuredURLLoader
from dotenv import load_dotenv
import os

# Carica variabili d'ambiente dal file .env
load_dotenv()

# Configurazione Streamlit
st.set_page_config(page_title="Riassunto notizie da Sito Web", page_icon="🦜")
st.title("🦜 Riassunto notizie da Sito Web")

# Chiave API Groq
groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model="llama3-70b-8192",
    max_tokens=1024,
    temperature=0.5
)

# Prompt Template
prompt_template = """
Provide a summary of the following article in 100 words and do not add personal comment.
Write the date of the article in the first sentence.
Here it is the article:
Content: {text}
"""


prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
generic_url=st.text_input("URL",label_visibility="collapsed")
if st.button("Summarize the Content from Website"):
    ## Validate all the inputs
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide the information to get started")

    else:
        try:
            with st.spinner("Waiting..."):
                ## loading the website
                loader=UnstructuredURLLoader(urls=[generic_url],
                                             ssl_verify=False,
                                             headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"})
                docs=loader.load()

                ## Chain For Summarization
                chain=load_summarize_chain(llm,chain_type="stuff",prompt=prompt)
                output_summary=chain.invoke(docs)

                st.success(output_summary)
        except Exception as e:
            st.exception(f"Exception:{e}")