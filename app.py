import streamlit as st
from core.retrieval import ask_agent

st.title("ChatGPT-like clone")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_container = st.empty()
        full_response = ""

        for chunk in ask_agent(prompt, stream=True):
            full_response += chunk
            response_container.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})