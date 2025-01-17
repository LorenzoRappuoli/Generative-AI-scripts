from crewai import Crew,Process
from cai_agents import agentFA,analysis_agent
from cai_tasks import research_task,reporting_task

# Forming the tech-focused crew with some enhanced configurations
crew = Crew(
  agents=[agentFA,analysis_agent],
  tasks=[research_task,reporting_task],
  process=Process.sequential,
  memory=True,
  cache=True,
  max_rpm=100,
  share_crew=True
)

## start the task execution process with enhanced feedback
result=crew.kickoff()
print(result)