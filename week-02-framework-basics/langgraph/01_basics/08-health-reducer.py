# Health Monitoring Dashboard with Status Label
# Demonstrates reducer usage with operator.add

import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph


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
    return {"risk_score": 0.3}


def classify_health(state: HealthState):
    score = state["risk_score"]

    if score < 0.8:
        status = "Normal"
    elif score < 1.2:
        status = "Moderate Risk"
    else:
        status = "High Risk"

    return {"health_status": status}


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

graph = graph_builder.compile()

initial_state = {"patient_name": "John Doe"}
final_state = graph.invoke(initial_state)

print(f"Patient Name: {final_state['patient_name']}")
print(f"Health risk score: {final_state['risk_score']:.2f}")
print(f"Health Status: {final_state['health_status']}")
