import streamlit as st
import requests


def query_backend(product_name, product_features, target_audience):
    url = f"http://127.0.0.1:8000/generate/marketing-email/"
    params = {
        "product_name": product_name,
        "product_features": product_features,
        "target_audience": target_audience
    }
    try:
        response = requests.get(url, params)
        if response.status_code == 200:
            return response.json().get("generated_text", "Error: No text generated.")
        else:
            return f"Error: {response.status_code}, {response.text}"
    except requests.exceptions.RequestException as e:
        return f"Request failed: {e}"

st.title("Generate a marketing email")

# Chat input
product_name = st.text_input("Enter your product name:")
product_features = st.text_input("Enter your product features:")
target_audience = st.text_input("Enter your target audience:")
if st.button("Generate") and product_name and product_features and target_audience:
    generated_json = query_backend(product_name, product_features, target_audience)
    st.json(generated_json)