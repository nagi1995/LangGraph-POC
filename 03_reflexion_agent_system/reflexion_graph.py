from typing import List
import json
from langchain_core.messages import BaseMessage, ToolMessage, message_to_dict
from langgraph.graph import END, MessageGraph

from chains import revisor_chain, first_responder_chain
from execute_tools import execute_tools
from dotenv import load_dotenv

load_dotenv()

graph = MessageGraph()
MAX_ITERATIONS = 1

def first_responder_chain_node(state):
    response = first_responder_chain.invoke(
        {
            "messages": state
        }
    )
    # print(f"response: {response}")
    return response

def revisor_chain_node(state):
    response = revisor_chain.invoke(
        {
            "messages": state
        }
    )
    # print(f"response: {response}")
    return response

graph.add_node("draft", first_responder_chain_node)
graph.add_node("execute_tools", execute_tools)
graph.add_node("revisor", revisor_chain_node)


graph.add_edge("draft", "execute_tools")
graph.add_edge("execute_tools", "revisor")

def event_loop(state: List[BaseMessage]) -> str:
    count_tool_visits = sum(isinstance(item, ToolMessage) for item in state)
    num_iterations = count_tool_visits
    if num_iterations > MAX_ITERATIONS:
        return END
    return "execute_tools"

graph.add_conditional_edges("revisor", event_loop)
graph.set_entry_point("draft")

app = graph.compile()

print(app.get_graph().draw_mermaid())

response = app.invoke(
    "Write about how small business can leverage AI to grow"
)

# print(response[-1].tool_calls[0]["args"]["answer"])

# for state in response:
#     print(f"\n\n{state}\n\n")

serialized_response = [message_to_dict(msg) for msg in response]
with open("./reflexion_response.json", "w") as f:
    json.dump(serialized_response, f, indent=4)
