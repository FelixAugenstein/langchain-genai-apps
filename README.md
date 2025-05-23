# LangChain GenAI Apps

Welcome to the **LangChain GenAI Apps** repository!

This project contains a collection of AI-powered backend and frontend applications built with **FastAPI**, **LangChain**, and **Streamlit**, designed for text generation use cases using **OpenAI** or **local Ollama models**.

---

## 🔧 Project Structure

### 📁 `fastapi-app/`

This folder contains the **FastAPI** backend with several endpoints showcasing generative AI capabilities:

- ✨ **Text Generation Endpoints**
  - General-purpose text generation using OpenAI or local Ollama models.
  - Custom prompts for specific use cases:
    - 🍽️ Recipe generation for a dish.
    - 📧 Marketing email generation.
    - 🧠 Therapy advice generation.
    - ✍️ Blog post creation.

- 🔎 **RAG (Retrieval-Augmented Generation) Examples**
  - Demonstrates how to use LangChain with RAG to generate responses based on custom document sources.

- 🤖 **Agent with Web Search**
  - Example endpoint that uses a LangChain agent with access to a web search tool.

### 📁 `streamlit-app/`

This folder contains the **Streamlit** frontend apps for the use cases supported in the backend.

Each UI corresponds to a specific generation task and provides an interactive interface to test and explore the models via the FastAPI endpoints.

---

## 🚀 Getting Started

Navigate to each folder for setup instructions:

- [`fastapi-app/README.md`](./fastapi-app/README.md)
- [`streamlit-app/README.md`](./streamlit-app/README.md)

---

## 🧠 Built With

- [FastAPI](https://fastapi.tiangolo.com/)
- [Streamlit](https://streamlit.io/)
- [LangChain](https://www.langchain.com/)
- [OpenAI](https://openai.com/)
- [Ollama](https://ollama.com/)

---

Feel free to explore, customize, and extend the examples for your own GenAI projects!