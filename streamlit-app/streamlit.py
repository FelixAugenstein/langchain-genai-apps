import streamlit as st
import requests


def query_backend(prompt):
    url = f"http://127.0.0.1:8000/generate/ollama/mistral/{prompt}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json().get("generated_text", "Error: No text generated.")
        else:
            return f"Error: {response.status_code}, {response.text}"
    except requests.exceptions.RequestException as e:
        return f"Request failed: {e}"

st.title("Chat with FastAPI Backend")

# Chat input
user_input = st.text_input("Enter your prompt:")
if st.button("Generate") and user_input:
    generated_text = query_backend(user_input)
    st.text_area("Generated Response:", generated_text, height=200)