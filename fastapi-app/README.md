# GenAI FastAPI App

### Steps to set up FastAPI app:

- Go to fastapi-app folder: `cd fastapi-app`
- make sure you have poetry installed: `curl -sSL https://install.python-poetry.org | python3 -`
- Set up virtualsenvs `poetry config virtualenvs.in-project true`
- Install dependencies `poetry install `
- Run the app: `poetry run uvicorn src.main:app --reload`

### How to create a new poetry project

```
mkdir fastapi-app
cd fastapi-app
poetry init --no-interaction
```
- install FastAPI and Uvicorn:
`poetry add fastapi uvicorn`
- set up directory structure:
```
mkdir src
touch src/main.py
```
- Run the app: `poetry run uvicorn src.main:app --reload`

Now you can access the FastAPI app at `http://127.0.0.1:8000`, and the Swagger UI should be available at `http://127.0.0.1:8000/docs`.

### Adding dependencies to Poetry:

Run the following commands:
```
poetry add python-dotenv
poetry add langchain
poetry add openai
```

### Working with Ollama

These are the steps to work with gemma-2b model and Ollama:

- Pull the model: `ollama pull gemma:2b`
- Check if Ollama is running: `ollama list`
- Try running: `ollama run gemma:2b "Hello"`
- Ensure FastAPI and LangChain are correctly installed: `pip show langchain langchain-community`


