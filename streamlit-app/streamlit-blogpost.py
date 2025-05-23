import streamlit as st
import requests


def query_backend(topic):
    url = f"http://127.0.0.1:8000/generate/blogpost/"
    params = {
        "topic": topic
    }
    try:
        response = requests.get(url, params)
        if response.status_code == 200:
            return response.json().get("generated_text", "Error: No text generated.")
        else:
            return f"Error: {response.status_code}, {response.text}"
    except requests.exceptions.RequestException as e:
        return f"Request failed: {e}"

st.title("Generate a Blogpost")

# Chat input
topic = st.text_input("Enter your topic:")
if st.button("Generate") and topic:
    generated_text = query_backend(topic)
    st.text_area("Generated Response:", generated_text, height=500)