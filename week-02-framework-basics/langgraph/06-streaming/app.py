import streamlit as st
from langchain_core.messages import HumanMessage
from graph_backend import graph, config

# Page Config
st.set_page_config(
    page_title="LangGraph Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 LangGraph Chatbot")
st.caption("A Streamlit UI on top of a LangGraph chatbot with tools and memory")

# Session State (UI Chat History)
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat Input
user_input = st.chat_input("Ask me anything...")

if user_input:
    # User message
    st.session_state.messages.append({"role": "user", "content": user_input} )
    
    with st.chat_message("user"):
        st.markdown(user_input)

    # Call LangGraph
    with st.spinner("Thinking..."):
        final_state = graph.invoke({"messages": [HumanMessage(content=user_input)]},config=config)

    assistant_reply = final_state["messages"][-1].content

    # Assistant message
    st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
    
    with st.chat_message("assistant"):
        st.markdown(assistant_reply)