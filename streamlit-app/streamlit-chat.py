import streamlit as st
import requests

st.title("Chat with AI")
st.markdown("### Powered by FastAPI and LangChain Memory")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
chat_container = st.container()
with chat_container:
    for chat in st.session_state.chat_history:
        st.markdown(f"**User:** {chat['user']}")
        st.markdown(f"**AI:** {chat['ai']}")
        st.markdown("---")

# User input
user_input = st.text_input("Type a message:", key="user_input")
if st.button("Send") and user_input:
    response = requests.post("http://127.0.0.1:8000/chat", json={"message": user_input}).json()
    st.session_state.chat_history.append({"user": user_input, "ai": response["response"]})
    st.experimental_rerun()