# Level 1: Pure Multi-Agent (create_react_agent + Supervisor)

# Goal: Get comfortable creating two distinct worker agents with their own tools, plus a supervisor routing between them.

# 🎯 Problem 1: The "Research & Fact-Checker" Team
# Scenario: When an agent researches a topic, it might make up facts. Build a team of two specialists:
# researcher: Has a search_notes tool to look up raw information.
# fact_checker: Has a verify_fact tool to cross-reference statements against a trusted knowledge base.

# Architecture:
# text
# START ──► [ supervisor ] ◄──────────────────────┐
#                  │                              │
#          ┌───────┴────────┐                     │
#          ▼                ▼                     │
#    [researcher]    [fact_checker]               │
#          │                │                     │
#          └────────────────┴─────────────────────┘
#                  │
#               "FINISH" ──► END

# Tools to write:
# @tool def search_notes(query: str) -> str
# @tool def verify_fact(claim: str) -> str

# What you learn:
# Creating two separate create_react_agent instances with dedicated tools.
# Writing custom wrapper nodes that append attributed messages (AIMessage(..., name="researcher")).
# Structured routing using Pydantic Literal["researcher", "fact_checker", "FINISH"].

from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.tools import tool
from langgraph.graph import START, StateGraph, MessagesState, END
from typing import Literal
from pydantic import BaseModel


@tool
def search_notes(query:str)->str:
    """ search here to know about some of the greatest cricketers"""
    cricketers = {
        "Virat Kohli": (
            """Virat Kohli is an Indian international cricketer widely regarded as one of the greatest batters in cricket history. 
            Born on November 5, 1988, in Delhi, he rose to prominence after captaining India to victory in the 2008 Under-19 World Cup. 
            Known as the "Run Machine," he holds the world record for the most centuries in One Day Internationals and has played key roles in winning the 2011 ODI World Cup, 
            the 2013 and 2025 Champions Trophies, and the 2024 T20 World Cup. 
            Having retired from T20 Internationals and Test cricket, he continues to represent India in ODIs 
            while playing for Royal Challengers Bengaluru in the IPL. 
            Celebrated for his fitness and aggressive leadership, 
            he has been honored with India's highest sporting award, the Major Dhyan Chand Khel Ratna, 
            alongside the Padma Shri and Arjuna Award."""
        ),
        "Rohit Sharma": (
            """Rohit Sharma is an Indian international cricketer celebrated as one of the greatest white-ball openers and captains in modern cricket. 
            Born on April 30, 1987, in Nagpur, he rose to global fame after debuting in 2007 and later transforming into an explosive opening batter. 
            Known as the "Hitman," he holds the world record for the highest individual score in an ODI (264), the most ODI double centuries (three), and the most sixes in international cricket. 
            Under his calm captaincy, India ended its ICC trophy drought by winning the 2024 T20 World Cup and the 2025 Champions Trophy, 
            alongside his five IPL titles as Mumbai Indians skipper. Having retired from T20 Internationals and Test cricket, 
            he continues to lead India in ODIs and play for Mumbai Indians in the IPL, 
            and he has been honored with the Major Dhyan Chand Khel Ratna and the Arjuna Award"""
        ),
        "Mahendra Singh Dhoni":(
            """Mahendra Singh Dhoni is a legendary former Indian cricketer widely regarded as one of the greatest captains and finishers in cricket history. 
            Born on July 7, 1981, in Ranchi, he rose from a small-town background to redefine Indian white-ball cricket with his sharp wicketkeeping and fearless lower-order batting. 
            Known as "Captain Cool," he remains the only captain in history to win all three major ICC white-ball trophies—the 2007 T20 World Cup, 
            the 2011 ODI World Cup, and the 2013 Champions Trophy—while also leading Chennai Super Kings to five IPL titles. 
            Having retired from international cricket in 2020, his storied association with Chennai Super Kings in the IPL has spanned nearly two decades, 
            though injuries recently kept him sidelined during the 2026 season. 
            Revered for his unmatched composure under pressure, he has been honored with India's highest sporting and civilian awards, 
            including the Major Dhyan Chand Khel Ratna, Padma Shri, and Padma Bhushan"""
        )
    }

    query_lower = query.lower()
    for name, details in cricketers.items():
        if name.lower() in query_lower or any(word.lower() in query_lower for word in name.split()):
            return f"Search results for '{name}':\n{details}"
            
    return "No information found for this cricketer in local notes."

@tool
def verify_fact(claim: str) -> str:
    """Checks whether a specific cricket claim is TRUE, FALSE, or UNVERIFIED."""
    claim_lower = claim.lower()
    
    VERIFIED_DATABASE = [
        {
            "keywords": ["kohli", "50", "century"],
            "status": "VERIFIED TRUE",
            "explanation": "Virat Kohli holds the record for 50 ODI centuries, reaching it in the 2023 World Cup."
        },
        {
            "keywords": ["dhoni", "3", "icc"],
            "status": "VERIFIED TRUE",
            "explanation": "Correct. MS Dhoni has won 3 ICC white-ball trophies."
        },
        {
            "keywords": ["rohit", "3", "double"],
            "status": "VERIFIED TRUE",
            "explanation": "Correct. Rohit Sharma is the only batter with 3 ODI double centuries."
        }
    ]
    
    for item in VERIFIED_DATABASE:
        # If all keywords appear in the claim
        if all(kw in claim_lower for kw in item["keywords"]):
            return f"[{item['status']}] {item['explanation']}"
            
    return f"[UNVERIFIED] Claim '{claim}' could not be verified in the official database."

llm = ChatOpenAI(model = "gpt-4o-mini",temperature=0)

researcher_agent = create_agent(
    model = llm,
    tools = [search_notes],
    system_prompt="youre a helpful assistant. use the search tool to retrieve the result that user asks"
)

fact_checker_agent = create_agent(
    model = llm,
    tools = [verify_fact],
    system_prompt="You are a strict fact-checker. Look at the conversation history and extract "
        "concrete, quantifiable claims (such as centuries, trophies, records, numbers). "
        "Use the verify_fact tool to verify them."
)

class AgentState(MessagesState):
    next:str

class Decision(BaseModel):
    next: Literal["researcher","fact_checker", "FINISH"]

supervisor_llm = llm.with_structured_output(Decision)

SUPERVISOR_SYSTEM_PROMPT = (
    "You are a supervisor managing a team of two specialist agents:\n"
    "\n"
    "  - researcher_agent : Finds facts, searches for information, gathers data\n"
    "  - fact_checker_agent   : Verifies the claim whether it is true or not.\n"
    "\n"
    "Given the conversation history, decide which agent should act next.\n"
    "\n"
    "Rules:\n"
    "- If the user request involves finding information, route to 'researcher' first\n"
    "- If research has been done and for claim route to 'fact_checker'\n"
    "- If both research and writing are complete, return 'FINISH'\n"
    "- If the task can be answered without research or fact checking, return 'FINISH'\n"
    "\n"
    "Respond with ONLY the name of the next agent, or 'FINISH'."
)

def supervisor(state:AgentState) -> dict:
    messages = [SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT)] + state['messages']

    decision:Decision = supervisor_llm.invoke(messages)
    return {
        "next": decision.next
    }

def researcher(state:AgentState)->dict:
    result = researcher_agent.invoke({"messages": state['messages']})
    last_message = result["messages"][-1]
    agent_message = AIMessage(content=last_message.content, name = "researcher")
    return {
        "messages": [agent_message]
    }

def fact_checker(state:AgentState)->dict:
    result = fact_checker_agent.invoke({"messages": state['messages']})
    last_message = result["messages"][-1]
    agent_message = AIMessage(content=last_message.content, name = "fact_checker")
    return {
        "messages":[agent_message]
    }

def route(state:AgentState) -> Literal["researcher","fact_checker","__end__"]:
    next_agent = state['next']

    if next_agent == "FINISH":
        return END
    return next_agent

builder = StateGraph(AgentState)
builder.add_node("supervisor",supervisor)
builder.add_node("researcher",researcher)
builder.add_node("fact_checker",fact_checker)

builder.add_edge(START,"supervisor")
builder.add_conditional_edges("supervisor",route)
builder.add_edge("researcher","supervisor")
builder.add_edge("fact_checker","supervisor")

multigraph = builder.compile()


initial_input = {
    "messages": [HumanMessage(content=(
        "Please research about virat kohli and then verify the claim."
    ))]
}

result = multigraph.invoke(initial_input)
for msg in result["messages"]:
    msg.pretty_print()