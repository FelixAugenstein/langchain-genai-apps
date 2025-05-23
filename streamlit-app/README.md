# Streamlit UI App

### Steps to set up Streamlit UI App:

- Go to streamlit-app folder: `cd streamlit-app`
- make sure you have poetry installed: `curl -sSL https://install.python-poetry.org | python3 -`
- Set up virtualsenvs `poetry config virtualenvs.in-project true`
- Install dependencies `poetry install --no-root`
- Run the app: `streamlit run <name_of_streamlit_app.py>` such as `streamlit run streamlit-recipe.py`

### Create new streamlit projects/UIs

- create new project:
```
mkdir streamlit-app
cd streamlit-app
poetry init --no-interaction
```
- install Streamlit:
`poetry add streamlit  `
- set up directory structure:
```
touch src/streamlit.py
```
- Run the app: `streamlit run streamlit.py`

Now the streamlit UI should open under `http://localhost:8501`

### Adding dependencies to Poetry:

Run the following commands:
```
poetry add uuid
```
