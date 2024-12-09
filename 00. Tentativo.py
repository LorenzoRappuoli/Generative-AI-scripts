from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import Html2TextTransformer
from scrapegraphai.utils.cleanup_html import cleanup_html
from scrapegraphai.nodes import SearchLinkNode
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import AzureChatOpenAI
import pandas as pd
from HTML_manager import HtmlManager
from langchain_core.output_parsers import JsonOutputParser
import json
import re

filename = "config.json"
with open(filename, 'r', encoding='utf-8-sig') as f:
    data = json.load(f)

AZURE_API_KEY = data['AZURE_API_KEY']
AZURE_ENDPOINT = data['AZURE_ENDPOINT']
DEPLOYMENT_NAME = data['DEPLOYMENT_NAME']
API_VERSION = data['API_VERSION']
MAX_TOKENS = data["MAX_TOKENS"]


_model = AzureChatOpenAI(
    api_key=AZURE_API_KEY,
    # temperature=0,
    api_version=API_VERSION,
    azure_endpoint=AZURE_ENDPOINT,
    deployment_name=DEPLOYMENT_NAME
)


urls = ["https://www.ricerca24.ilsole24ore.com/?cmd=static&chId=30&path=/search/search_engine.jsp&field=Titolo|Testo&orderBy=score+desc&chId=30&disable_user_rqq=false&keyWords=Leonardo&pageNumber=1&pageSize=10&fromDate=&toDate=&filter=all"]
loader = AsyncChromiumLoader(urls,
                             )
docs = loader.load()

#html2text = Html2TextTransformer(ignore_links = False)
#docs_transformed = html2text.transform_documents(docs)
#print(docs_transformed[0])

print("Caricati tutti i link nelle pagine")

### ------ da inserire blocco per evitare di raccogliere gli stessi link già visitati ----------------------

source = "https://www.ricerca24.ilsole24ore.com/?cmd=static&chId=30&path=/search/search_engine.jsp&field=Titolo|Testo&orderBy=score+desc&chId=30&disable_user_rqq=false&keyWords=Leonardo&pageNumber=1&pageSize=10&fromDate=&toDate=&filter=all"

text = docs[0].page_content[:]

title, minimized_body, retrieved_links, image_urls = cleanup_html(text, base_url=source)

retrieved_links = list(dict.fromkeys(retrieved_links)) # remove duplicates

#search_link_node = SearchLinkNode(
#    input= "user_prompt & links",
#    output=["relevant_links"],
#    node_config={"llm_model": _model, "verbose": False}
#)
output_parser = JsonOutputParser()

#user_prompt = "articles and other information about the italian company Leonardo" # da rivedere
#
#prompt_relevant_links = """
#    You are a website scraper and you have just scraped the following links from a website: {links}
#    Now you have to follow the following prompt: From the following list of links, you have to filter
#    and extract relevant links that are most likely to contain {user_prompt}
#    Sort them in order of importance, the first one should be the most important one, the last one the least important.\n
#    The output must be a dict with key: "relevant_links", and the value must be the list of links.\n
#    The list of links must contain just links, not comment or other.
#    Output instructions: {format_instructions}.
#    """


#user_prompt = "articles and industry-related information about the company {company_name}, operating in the {industry} industry."
#
#prompt_relevant_links = """
#    You are a website scraper and have just scraped the following links from a website: {links}.
#    Your task is to filter and extract only the links that are most relevant to the following prompt: "{user_prompt}".
#    These links should specifically pertain to business, industry, or corporate news related to the company {company_name}, operating in the {industry} industry.
#    Avoid including links that may reference other entities, individuals, or topics unrelated to this specific company.
#    Sort the list in order of relevance, with the most relevant link first and the least relevant last.
#
#    The output should be a dictionary with the key "relevant_links" and a list of only the relevant links as the value, without any comments or additional content.
#    Output format instructions: {format_instructions}.
#    """

user_prompt = "articles and business or defense industry-related information about the Italian aerospace and defense company Leonardo S.p.A."

prompt_relevant_links = """
    You are a website scraper and have just scraped the following links from a website: {links}.
    Your task is to filter and extract only the links that are most relevant to the following prompt: "{user_prompt}". 
    These links should specifically pertain to business, defense industry, or corporate news related to the Italian company Leonardo S.p.A. 
    Do not include links that might reference other subjects or persons named Leonardo. Sort the list in order of relevance, with the most important link first and the least important last.
    The output should be a dictionary with the key "relevant_links" and a list of only the relevant links as the value, without any comments or extra content.
    Output format instructions: {format_instructions}.
    """

merge_prompt = PromptTemplate(
    template=prompt_relevant_links,
    input_variables=["links","user_prompt"],
    partial_variables = {"format_instructions": output_parser.get_format_instructions()}
)

merge_chain = merge_prompt|_model|output_parser

relevant_links = merge_chain.invoke({"links": retrieved_links, "user_prompt": user_prompt})

print("Link rilevanti: ")
print(relevant_links)
print("------------------------------------------------------------------------------")

links_tbs = []
for l in relevant_links['relevant_links']:
    links_tbs.append(l)

loader_1 = AsyncChromiumLoader(links_tbs,)
docs1 = loader_1.load()
data = []
dizionario = {}
indx = 0

for d in docs1:
    text = d.page_content[:]
    title, minimized_body, retrieved_links, image_urls = cleanup_html(text, base_url=links_tbs[indx])
    data.append({
       'title': title,
       'minimized_body': minimized_body,
       'retrieved_links': retrieved_links,
       'image_urls': image_urls
    })
    dizionario[links_tbs[indx]] = minimized_body
    indx = indx + 1

df = pd.DataFrame(data)
df.to_excel("Risultati dai link.xlsx")

## ripulisco i body delle notizie rilevanti

output_parser = JsonOutputParser()

# forse da aggiungere i nomi dei competitor?
# versione vecchia che rompe le policy di Azure
#prompt_chunks = """
#                You are a website scraper and you have just scraped a content from the following website that contains news: {link}.\n
#                You are now asked to retrieve the text of the news contained in the html, deleting all the non relevant info like cookies, etc...\n
#                The website is big so I am giving you one chunk at the time to be merged later with the other chunks.\n
#                Output instructions: {format_instructions}\n
#                ---CHUNK NUMBER {chunk_id} STARTS HERE:\n {context}.\n ---CHUNK {chunk_id} END HERE---. \n
#                """
#
#prompt_total = """
#                You are a website scraper and you have just scraped a content from a news website.\n
#                You are now asked to retrieve the text of the article of {question} that you have scraped, deleting all the non relevant info like cookies, etc...\n.
#                You have scraped many chunks since the website is big and now you are asked to
#                merge them into a single answer without repetitions (if there are any).\n
#                Make sure that if a maximum number of items is specified in the instructions
#                that you get that maximum number and do not exceed it. \n
#                The output must be "source", "country", "date (day/month/year)", "link", "title" and "body".
#                The body must NOT contain any quotes (" or '), so replace the quotes with <>.
#                If the page contains an article, retrieve the body of the article of the website; if the page contain just lists
#                of titles of other, write "NO BODY FOUND".
#                Output instructions: {format_instructions}\n
#                Website content: {context}\n """
#
#prompt_no_chunk = """
#                You are a website scraper and you have just scraped a content from a website.\n
#                You are now asked to to retrieve the text of the article of {question} that you have scraped, deleting all the non relevant info like cookies, etc...\n.
#                The output must be "source", "country", "date (day/month/year)", "link", "title" and "body".
#                The body must NOT contain any quotes (" or '), so replace the quotes with <>.
#                If the page contains an article, retrieve the body of the article of the website; if the page contain just lists
#                of titles of other, write "NO BODY FOUND".
#                Make sure that the links are correct and contain articles.
#                Output instructions: {format_instructions}\n
#                Website content: {context}\n """



# prompt_total = """
# As a web content summarizer, you have gathered sections from a news website and need to combine them into a single summary.\n
# Retrieve the main text of the article for {question}, excluding any extraneous information like cookie notifications.\n
# Ensure no repetition occurs when combining sections, and, if a maximum number of items is specified, adhere to that limit.\n
# The output should include "source", "country", "date (day/month/year)", "link", "title", and "body".\n
# In the body, substitute any quotes with <>. If the page contains an article body, include it; otherwise, write "NO BODY FOUND".\n
# Follow these output instructions: {format_instructions}\n
# Content collected: {context}\n
# """
#
# prompt_no_chunk = """
# As a web content summarizer, you have retrieved content from a news website.\n
# Extract the main text of the article for {question}, excluding irrelevant items like cookie notifications.\n
# The output should include "source", "country", "date (day/month/year)", "link", "title", and "body".\n
# In the body, replace any quotes with <>. If the page includes an article body, retrieve it; if it only lists titles, write "NO BODY FOUND".\n
# Ensure links are accurate and correspond to article content.\n
# Output format: {format_instructions}\n
# Website content provided: {context}\n
# """

prompt_total = """
As a web content summarizer, you have gathered sections from a news website and need to combine them into a single summary.\n
Retrieve the main text of the article for {question}, excluding any extraneous information like cookie notifications.\n 
Ensure no repetition occurs when combining sections, and, if a maximum number of items is specified, adhere to that limit.\n
The output should include "source", "country", "date (day/month/year)", "link", "title", and "body".\n
In "date" you should write the exact date when the article was writen. \n
In the body, substitute any quotes with <>. If the page contains an article body, include it; otherwise, write "NO BODY FOUND".\n
Follow these output instructions: {format_instructions}\n
Content collected: {context}\n
"""
prompt_no_chunk = """
As a web content summarizer, you have retrieved content from a news website.\n
Extract the main text of the article for {question}, excluding irrelevant items like cookie notifications.\n 
The output should include "source", "country", "date (day/month/year)", "link", "title", and "body".\n
in "date" you should write the exact date when the article was writen. \n
In the "body", substitute any quotes with <>. If the page contains an article body, include it; otherwise, write "NO BODY FOUND".\n
Ensure links are accurate and correspond to article content.\n
Output format: {format_instructions}\n
Website content provided: {context}\n
"""

prompt_chunks = """
As a web content summarizer, you have scraped content from a news website at this link: {link}.\n
Please retrieve the main text of the article from this HTML snippet, omitting any irrelevant information, such as cookie notifications.\n 
Since the website content is large, you’ll be processing it one section at a time, and the sections will be combined later.\n
Output format should follow these instructions: {format_instructions}\n
---START OF SECTION {chunk_id}---\n {context}\n---END OF SECTION {chunk_id}---\n
"""

result_dict = {}
chains_dict = {}
links = []
answers = []
print("---------------------------------------Inizio run---------------------------------------")
for link, html in dizionario.items():


    links.append(link)
    # Estrarre il testo dalla pagina HTML

    text = HtmlManager().get_text(str(html))
    #print(text)

    # Dividere il testo in chunk basati sul limite di token
    text_chunk = HtmlManager().chunk_text(text, MAX_TOKENS)

    # Svuotare chains_dict per ogni nuovo link
    chains_dict = {}  # Nuovo dizionario per ogni link


    for i, chunks in enumerate(text_chunk):

        if len(text_chunk) > 1:
            # Prompt per i chunk multipli
            prompts = PromptTemplate(template=prompt_chunks,
                                     input_variables=["user_prompt"],
                                     partial_variables={
                                         "context": chunks,
                                         "chunk_id": i + 1,
                                         "link": link,
                                         "format_instructions": output_parser.get_format_instructions()})

            chain_name = f"chunk{i + 1}"
            chains_dict[chain_name] = prompts | _model

            # Esecuzione in parallelo delle catene
            map_chain = RunnableParallel(**chains_dict)
            answer = map_chain.invoke({"question": user_prompt})

            # Prompt di fusione
            merge_prompt = PromptTemplate(template=prompt_total,
                                          input_variables=["context"],
                                          partial_variables={
                                              "format_instructions": output_parser.get_format_instructions()})

            merge_chain = merge_prompt | _model
            answer = merge_chain.invoke({"context": answer, "question": user_prompt})
            answers.append(answer)
        else:
            # Prompt per singolo chunk
            merge_prompt = PromptTemplate(template=prompt_no_chunk,
                                          input_variables=["context"],
                                          partial_variables={
                                              "format_instructions": output_parser.get_format_instructions()})

            merge_chain = merge_prompt | _model
            answer = merge_chain.invoke({"context": chunks, "question": user_prompt})
            answers.append(answer)
            #answer.content
            #answer.json

print("------------------------------------Fine Run----------------------------------------------")


datajson = []

for an in answers:
    an = str(an)
    content = an.split("additional_kwargs")[0]

    # Usa regex per estrarre ogni campo
    source = re.search(r'"source": "(.*?)"', content)
    country = re.search(r'"country": "(.*?)"', content)
    date = re.search(r'"date": "(.*?)"', content)
    link = re.search(r'"link": "(.*?)"', content)
    title = re.search(r'"title": "(.*?)"', content)
    body = re.search(r'"body": "(.*?)"', content)

    # Appende i valori estratti o None nel caso non siano presenti
    datajson.append({
        "source": source.group(1) if source else None,
        "country": country.group(1) if country else None,
        "date": date.group(1) if date else None,
        "link": link.group(1) if link else None,
        "title": title.group(1) if title else None,
        "body": body.group(1) if body else None
    })

df_json = pd.DataFrame(datajson)
df_json.to_excel("df_json.xlsx")






