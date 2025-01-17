from crewai import Agent
from crewai_tools import ScrapeWebsiteTool, PDFSearchTool
from dotenv import load_dotenv
import os
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_MODEL_NAME"]="gpt-4-0125-preview"

toolws = ScrapeWebsiteTool(website_url='https://markets.ft.com/data/equities/tearsheet/summary?s=LDO:MIL')
toolpdf = PDFSearchTool(pdf='./Risultati.pdf')

# creazione agenti

agentFA = Agent(
    role="Senior Financial Analyst",
    goal="Analyze and interpret complex datasets to provide actionable insights about Leonardo Company",
    backstory="With over 10 years of experience in financial analysis, "
              "you excel at finding patterns in complex datasets.",
    llm="gpt-4",  # Default: OPENAI_MODEL_NAME or "gpt-4"
    function_calling_llm=None,  # Optional: Separate LLM for tool calling
    memory=True,  # Default: True
    verbose=False,  # Default: False
    allow_delegation=False,  # Default: False
    max_iter=5,  # Default: 20 iterations
    max_rpm=None,  # Optional: Rate limit for API calls
    max_execution_time=None,  # Optional: Maximum execution time in seconds
    max_retry_limit=2,  # Default: 2 retries on error
    allow_code_execution=False,  # Default: False
    code_execution_mode="safe",  # Default: "safe" (options: "safe", "unsafe")
    respect_context_window=True,  # Default: True
    use_system_prompt=True,  # Default: True
    tools=[toolws, toolpdf],  # Optional: List of tools
    knowledge_sources=None,  # Optional: List of knowledge sources
    embedder_config=None,  # Optional: Custom embedder configuration
    system_template=None,  # Optional: Custom system prompt template
    prompt_template=None,  # Optional: Custom prompt template
    response_template=None,  # Optional: Custom response template
    step_callback=None,  # Optional: Callback function for monitoring
)

analysis_agent = Agent(
    role="Leonardo Reporting Analyst",
    goal="Create detailed reports based on Leonardo Company'S data",
    backstory="You're a meticulous analyst with a keen eye for detail. You're known for"
              "your ability to turn complex data into clear and concise reports, making"
              "it easy for others to understand and act on the information you provide.",
    memory=True,
    respect_context_window=True,
    max_rpm=10,  # Limit API calls
    function_calling_llm="gpt-4o-mini"  # Cheaper model for tool calls
)