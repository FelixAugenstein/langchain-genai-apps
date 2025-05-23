import streamlit as st
import requests

# Config
API_URL = "http://localhost:8000/rag/vector-store-upload-doc"

# Session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# App UI
st.set_page_config(page_title="RAG Chat with PDF", page_icon="📄")
st.title("📄 Chat with Your PDF")

# Upload PDF
uploaded_pdf = st.file_uploader("Upload a PDF", type="pdf")

# Input question
user_question = st.text_input("Ask a question about the PDF", placeholder="Enter your question...")

# Ask button
if st.button("Ask") and uploaded_pdf and user_question:
    with st.spinner("Sending query..."):

        # Prepare files and data for the FastAPI endpoint
        files = {"file": (uploaded_pdf.name, uploaded_pdf.getvalue(), "application/pdf")}
        data = {"question": user_question}

        try:
            response = requests.post(API_URL, files=files, data=data)
            response.raise_for_status()
            answer = response.json().get("answer")

            # Save user message and bot answer
            st.session_state.messages.append({"role": "user", "content": user_question})
            st.session_state.messages.append({"role": "assistant", "content": answer})

        except Exception as e:
            st.error(f"Error: {e}")

# Display the chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    else:
        st.markdown(f"**AI:** {msg['content']}")
