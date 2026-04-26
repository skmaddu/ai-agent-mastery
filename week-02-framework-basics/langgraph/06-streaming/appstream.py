import streamlit as st
from langchain_core.messages import  HumanMessage, AIMessage, AIMessageChunk
from graph_backend import graph, config

# Page Config
st.set_page_config(
    page_title="LangGraph Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 LangGraph Chatbot")
st.caption("A Streamlit UI on top of a LangGraph chatbot with tools, memory and streaming")

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
    # User Message
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    with st.chat_message("user"):
        st.markdown(user_input)

    # Assistant response (streaming using a placeholder)
    with st.chat_message("assistant"):
        # Create an empty space in the UI
        placeholder = st.empty()
        full_response = ""

        # Stream output from LangGraph
        for message_chunk, metadata in graph.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
            stream_mode="messages"
         ):
            if isinstance(message_chunk, (AIMessage, AIMessageChunk)):
                if message_chunk.content:
                    full_response += message_chunk.content
                    placeholder.markdown(full_response)

    # Save assistant message to UI history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
