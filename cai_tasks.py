from crewai import Task
from cai_agents import agentFA,analysis_agent

research_task = Task(
    description="""
        Conduct a thorough research about Leonardo Company's status.
        Make sure you find any interesting and relevant information given
        the current year is 2025 and last year was 2024.
    """,
    expected_output="""
        A list with 10 bullet points of the most relevant information about Leonardo company's situation.
    """,
    agent=agentFA
)

reporting_task = Task(
    description="""
        Review the context you got and expand each topic into a full section for a report.
        Make sure the report is detailed and contains any and all relevant information.
    """,
    expected_output="""
        A fully fledge reports with the mains topics, each with a full section of information.
        Formatted as markdown without '```'
    """,
    agent=analysis_agent,
    context=[research_task],
    output_file="report.md"
)