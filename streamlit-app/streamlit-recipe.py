import streamlit as st
import requests


def query_backend(country, desired_dish, ingredients):
    url = f"http://127.0.0.1:8000/generate/recipe/"
    params = {
        "country": country,
        "desired_dish": desired_dish,
        "ingredients": ingredients
    }
    try:
        response = requests.get(url, params)
        if response.status_code == 200:
            return response.json().get("generated_text", "Error: No text generated.")
        else:
            return f"Error: {response.status_code}, {response.text}"
    except requests.exceptions.RequestException as e:
        return f"Request failed: {e}"

st.title("Generate a Recipe")

# Chat input
country = st.text_input("Enter your country:")
desired_dish = st.text_input("Enter your desired dish:")
ingredients = st.text_input("Enter your ingredients:")
if st.button("Generate") and country and desired_dish and ingredients:
    generated_text = query_backend(country, desired_dish, ingredients)
    st.text_area("Generated Response:", generated_text, height=500)