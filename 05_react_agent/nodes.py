import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from logger_config import logger

from dotenv import load_dotenv

from agent_reason_runnable import react_agent_runnable, tools
from react_state import AgentState

load_dotenv()

def reason_node(state: AgentState):
    logger.info(f"state: {state}")
    agent_outcome = react_agent_runnable.invoke(state)
    logger.info(f"agent_outcome: {agent_outcome}")
    return {"agent_outcome": agent_outcome}


def act_node(state: AgentState):
    logger.info(f"state: {state}")
    agent_action = state["agent_outcome"]
    
    # Extract tool name and input from AgentAction
    tool_name = agent_action.tool
    tool_input = agent_action.tool_input
    
    # Find the matching tool function
    tool_function = None
    for tool in tools:
        if tool.name == tool_name:
            tool_function = tool
            break
    
    # Execute the tool with the input
    if tool_function:
        if isinstance(tool_input, dict):
            output = tool_function.invoke(**tool_input)
        else:
            output = tool_function.invoke(tool_input)
    else:
        output = f"Tool '{tool_name}' not found"
    
    logger.info(f"intermediate_steps: {[(agent_action, str(output))]}")
    return {"intermediate_steps": [(agent_action, str(output))]}

