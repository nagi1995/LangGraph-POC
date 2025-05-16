import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from logger_config import logger



from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from typing import TypedDict

from dotenv import load_dotenv

load_dotenv()

class State(TypedDict):
    text: str

def node_a(state: State): 
    logger.info("Node A")
    return Command(
        goto="node_b", 
        update={
            "text": state["text"] + "a"
        }
    )

def node_b(state: State): 
    logger.info("Node B")
    return Command(
        goto="node_c", 
        update={
            "text": state["text"] + "b"
        }
    )


def node_c(state: State): 
    logger.info("Node C")
    return Command(
        goto=END, 
        update={
            "text": state["text"] + "c"
        }
    )

graph = StateGraph(State)

graph.add_node("node_a", node_a)
graph.add_node("node_b", node_b)
graph.add_node("node_c", node_c)

graph.set_entry_point("node_a")


app = graph.compile()

response = app.invoke({
    "text": ""
})

logger.info(response)

