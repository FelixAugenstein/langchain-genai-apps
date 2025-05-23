import streamlit as st
import requests


def query_backend(disease, pathogen):
    url = f"http://127.0.0.1:8000/generate/therapy/"
    params = {
        "disease": disease,
        "pathogen": pathogen
    }
    try:
        response = requests.get(url, params)
        if response.status_code == 200:
            return response.json().get("generated_text", "Error: No text generated.")
        else:
            return f"Error: {response.status_code}, {response.text}"
    except requests.exceptions.RequestException as e:
        return f"Request failed: {e}"

st.title("Generate a Therapy")

# Chat input
disease = st.text_input("Enter your disease:")
pathogen = st.text_input("Enter your pathogen:")
if st.button("Generate") and disease and pathogen:
    generated_text = query_backend(disease, pathogen)
    st.text_area("Generated Response:", generated_text, height=500)