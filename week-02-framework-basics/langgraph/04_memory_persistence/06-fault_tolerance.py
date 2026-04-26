# Fault-Tolerant Health Monitoring Workflow

from typing import Annotated, TypedDict
import operator
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
import time

class HealthState(TypedDict):
    patient_name: str
    risk_score: Annotated[float, operator.add]
    health_status: str

def heart_rate_monitor(state: HealthState):
    print("Checking heart rate...")
    return {"risk_score": 0.4}

def oxygen_monitor(state: HealthState):
    print("Checking oxygen level...")
    return {"risk_score": 0.7}

def temperature_monitor(state: HealthState):
    print("Checking body temperature...")
    time.sleep(10) 
    return {"risk_score": 0.3}
 
def classify_health(state: HealthState):
    score = state["risk_score"]
    print("Getting health status...")
    if score < 0.8:
        status = "Normal"
    elif score < 1.2:
        status = "Moderate Risk"
    else:
        status = "High Risk"

    return { "health_status": status }
   
graph_builder = StateGraph(HealthState)

graph_builder.add_node("heart_rate_monitor", heart_rate_monitor)
graph_builder.add_node("oxygen_monitor", oxygen_monitor)
graph_builder.add_node("temperature_monitor", temperature_monitor)
graph_builder.add_node("classify_health", classify_health)

graph_builder.add_edge(START, "heart_rate_monitor")
graph_builder.add_edge("heart_rate_monitor", "oxygen_monitor")
graph_builder.add_edge("oxygen_monitor", "temperature_monitor")
graph_builder.add_edge("temperature_monitor", "classify_health")
graph_builder.add_edge("classify_health", END)

conn = sqlite3.connect("health_checkpoints.sqlite", check_same_thread=False)
checkpointer = SqliteSaver(conn)
graph = graph_builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "patient-001"}}

initial_state = {"patient_name": "Tim Dave", "risk_score": 0.0}

final_state = graph.invoke(initial_state,config=config)
#final_state = graph.invoke(None,config=config)
print(f"Patient Name: {final_state['patient_name']}")
print(f"Health risk score: {final_state['risk_score']:.2f}")
print(f"Health Status: {final_state['health_status']}")
