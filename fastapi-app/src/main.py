import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional  # Add this import for Optional
#from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOllama

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain


# Load environment variables from .env file
load_dotenv()

# Access your OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize FastAPI app
app = FastAPI(
    title="GenAI FastAPI App",
    description="This is a FastAPI application that demonstrates API endpoints and LangChain integration."
)

# Example GET endpoint
@app.get("/")
def read_root():
    return {"message": "GenAI FastAPI App. To see the swagger UI go to /docs"}

# Endpoint to generate text using OpenAI (via LangChain)
@app.get("/generate/openai/{prompt}", tags=["Generation"])
def generate_text(prompt: str):
    if not openai_api_key:
        return {"error": "OpenAI API key is missing"}

    # Initialize OpenAI LLM with LangChain
    llm = OpenAI(openai_api_key=openai_api_key)

    # Use LangChain to process the prompt and generate text
    prompt_template = PromptTemplate(input_variables=["prompt"], template="{prompt}")
    chain = LLMChain(llm=llm, prompt=prompt_template)
    result = chain.run(prompt=prompt)

    return {"prompt": prompt, "generated_text": result}


# Endpoint to generate text using Ollama gemma:2b (via LangChain)
@app.get("/generate/ollama/gemma2b/{prompt}", tags=["Generation"])
def generate_text(prompt: str):
    llm=ChatOllama(model="gemma:2b", base_url="http://localhost:11434")

    response = llm.invoke(prompt)
    result = response.content

    return {"prompt": prompt, "generated_text": result}


# Endpoint to generate text using Ollama gemma:2b (via LangChain)
@app.get("/generate/ollama/mistral/{prompt}", tags=["Generation"])
def generate_text(prompt: str):
    llm=ChatOllama(model="mistral", base_url="http://localhost:11434")

    response = llm.invoke(prompt)
    result = response.content

    return {"prompt": prompt, "generated_text": result}


# Endpoint to generate text using Ollama gemma:2b (via LangChain)
@app.get("/generate/ollama/llama32/{prompt}", tags=["Generation"])
def generate_text(prompt: str):
    llm=ChatOllama(model="llama3.2", base_url="http://localhost:11434")

    response = llm.invoke(prompt)
    result = response.content

    return {"prompt": prompt, "generated_text": result}


# Endpoint to generate recipe (via LangChain)
@app.get("/generate/recipe", tags=["Recipe"])
def generate_text(country: str, desired_dish: str, ingredients: str):
    llm=ChatOllama(model="mistral", base_url="http://localhost:11434")

    prompt_template = PromptTemplate(
        input_variables=["country","desired_dish","ingredients"],
        template="""You are an expert in traditional cuisines.
        You provide information about a specific dish from a specific country.
        Avoid giving information about fictional places. If the country is fictional
        or non-existent answer: I don't know.
        Answer the question: What is the traditional cuisine of {country}?
        Then you should also describe a recipe {desired_dish} using the following ingredients: {ingredients}.
        """
    )

    response = llm.invoke(prompt_template.format(country=country,
                                                 desired_dish=desired_dish,
                                                 ingredients=ingredients
                                                 ))

    result = response.content

    return {"prompt_template": prompt_template, "generated_text": result}


# Endpoint to generate therapy (via LangChain)
@app.get("/generate/therapy", tags=["Therapy"])
def generate_text(disease: str, pathogen: str):
    llm=ChatOllama(model="deepseek-r1", base_url="http://localhost:11434")

    prompt_template = PromptTemplate(
        input_variables=["disease","pathogen"],
        template="""Du bist ein Experte und unterstützt Ärzte.
        Du erhältst Informationen von einem Patienten, welche Krankheit und Erreger der Patient hat.
        Du sollst Therapieoptionen vorschlagen und bei medikamentöser Therapie 
        genaue Medikamentennamen und wie diese Medikamente eingenommen werden sollen. 
        Gib keine fiktionalen Informationen und Antworten, gib nur die Wahrheit aus,
        wenn du etwas nicht weißt, antworte mit: Das weiß ich leider nicht.
        Du sollst außerdem immer auf Deutsch antworten.
        Die Krankheit ist: {disease}.
        Der Erreger ist: {pathogen}.
        Antwort: 
        """
    )

    response = llm.invoke(prompt_template.format(disease=disease,
                                                 pathogen=pathogen
                                                 ))

    result = response.content

    return {"prompt_template": prompt_template, "generated_text": result}


# Endpoint to generate blogpost (via LangChain)
@app.get("/generate/blogpost", tags=["Blogpost"])
def generate_text(topic: str):
    llm=ChatOllama(model="deepseek-r1", base_url="http://localhost:11434")

    outline_prompt = PromptTemplate(
        input_variables=["topic"],
        template="""Your are professional blogger.
        Create an outline for a blog post on the following topic: {topic}
        The outline should include:
        - Intoduction
        - 3main points with subpoints
        - Conslusion
        """
    )

    introduction_prompt = PromptTemplate(
        input_variables=["outline"],
        template="""You are a professional blogger.
        Write an engaging introduction paragraph based on the following outline: {outline}
        The introduction should hook the reader and provide a brief overview of the topic.
        """
    )

    first_chain = outline_prompt | llm | StrOutputParser() #| (lambda title: (st.write(title),title)[1])
    second_chain = introduction_prompt | llm
    overall_chain = first_chain | second_chain

    response = overall_chain.invoke({"topic":topic})

    result = response.content

    return {"outline_prompt": outline_prompt, "introduction_prompt": introduction_prompt, "generated_text": result}



# Endpoint to generate marketing email (via LangChain)
@app.get("/generate/marketing-email", tags=["Marketing email"])
def generate_text(product_name: str, product_features: str, target_audience: str):
    llm=ChatOllama(model="deepseek-r1", base_url="http://localhost:11434")

    product_prompt = PromptTemplate(
        input_variables=["product_name", "product_features"],
        template="""You are an experienced marketing specialist.
        Create a catchy subject line for a marketing email promoting the following product: 
        {product_name}.
        Highlight these features: {product_features}
        Respond with only the subject line.
        """
    )

    email_prompt = PromptTemplate(
        input_variables=["subject_line", "product_name", "target_audience"],
        template="""Write a marketing email of 300 words for the product {product_name}
        Use the subject line: {subject_line}
        Tailor the message for the following target audience: {target_audience}
        Format the output as a JSON object with three keys: 'subject', 'audience', 'email'
        and fill them with the respective values.
        """
    )

    first_chain = product_prompt | llm | StrOutputParser()
    second_chain = email_prompt | llm | JsonOutputParser()
    overall_chain = (
        first_chain |
        (lambda subject_line: {"subject_line": subject_line, "product_name": product_name, "target_audience": target_audience}) |
        second_chain
    )

    response = overall_chain.invoke({"product_name": product_name, "product_features": product_features})

    print(response)

    return {"generated_text": response}



# Initialize memory storage
memory = ConversationBufferMemory()

class ChatRequest(BaseModel):
    message: str

@app.post("/chat", tags=["Chat with Session History"])
async def chat_endpoint(request: ChatRequest):
    user_message = request.message
    memory.save_context({"input": user_message}, {"output": ""})
    chat = ChatOllama(model="deepseek-r1", base_url="http://localhost:11434")
    response = chat.predict(memory.load_memory_variables({})["history"])
    memory.save_context({"input": user_message}, {"output": response})
    return {"response": response, "history": memory.load_memory_variables({})["history"]}


from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from pathlib import Path
import tempfile
from typing import Annotated
from fastapi import File, UploadFile, Form
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain

# Endpoints for RAG (via LangChain)
@app.get("/rag/vector-store", tags=["RAG"])
def embed_and_chat_with_doc(question: str):

    # Initialize LLM and embeddings
    llm=ChatOllama(model="deepseek-r1", base_url="http://localhost:11434")
    #embeddings = OllamaEmbeddings(model_name="llama3:latest")
    embeddings = OllamaEmbeddings(model="llama3")

    # Load and process the document
    base_dir = Path(__file__).resolve().parent
    pdf_path = base_dir / "data" / "academic_research_data.pdf"
    document = PyPDFLoader(str(pdf_path)).load() # instead of TextLoader
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(document)

    # Create vector store
    vector_store = Chroma.from_documents(chunks,embeddings)
    retriever = vector_store.as_retriever()

    # Prompt and chains
    prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system","""You are an assistant for answering questions.
        Use the provided context to respond.If the answer 
        isn't clear, acknowledge that you don't know. 
        Limit your response to three concise sentences.
        {context}
        
        """),
        ("human", "{input}")
    ]
    )

    history_aware_retriever = create_history_aware_retriever(llm,retriever,prompt_template)
    qa_chain = create_stuff_documents_chain(llm,prompt_template)
    rag_chain = create_retrieval_chain(history_aware_retriever,qa_chain)

    if question:
        response = rag_chain.invoke({"input":question})
        print(response['answer'])
        return {"answer": response['answer']}
    

@app.post("/rag/vector-store-upload-doc", tags=["RAG"])
async def embed_and_chat_with_doc(
    question: Annotated[str, Form(...)],
    file: Annotated[UploadFile, File(...)]
):

    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(await file.read())  # Save uploaded file content
        temp_pdf_path = Path(temp_pdf.name)  # Get the file path to parse the PDF


    # Initialize LLM and embeddings
    llm=ChatOllama(model="deepseek-r1", base_url="http://localhost:11434")
    #embeddings = OllamaEmbeddings(model_name="llama3:latest")
    embeddings = OllamaEmbeddings(model="llama3")

    # Load and process the document
    document = PyPDFLoader(str(temp_pdf_path)).load() # instead of TextLoader
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(document)

    # Create vector store
    vector_store = Chroma.from_documents(chunks,embeddings)
    retriever = vector_store.as_retriever()

    # Prompt and chains
    prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system","""You are an assistant for answering questions.
        Use the provided context to respond.If the answer 
        isn't clear, acknowledge that you don't know. 
        Limit your response to three concise sentences.
        {context}
        
        """),
        ("human", "{input}")
    ]
    )

    history_aware_retriever = create_history_aware_retriever(llm,retriever,prompt_template)
    qa_chain = create_stuff_documents_chain(llm,prompt_template)
    rag_chain = create_retrieval_chain(history_aware_retriever,qa_chain)

    if question:
        response = rag_chain.invoke({"input":question})
        print(response['answer'])
        return {"answer": response['answer']}

from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchRun
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain.tools import Tool
from langchain.utilities import WikipediaAPIWrapper, DuckDuckGoSearchAPIWrapper

# Wikipedia tool settings
import wikipedia
from bs4 import BeautifulSoup

# Patch Wikipedia's use of BeautifulSoup to use lxml
def patched_beautifulsoup(html):
    return BeautifulSoup(html, features="lxml")

@app.get("/agent/web-search", tags=["Agents"])
def web_search(question: str):

     # LLM - Ollama
    llm = ChatOllama(model="llama3", base_url="http://localhost:11434")

    # Tools - wrapped for langchain
    wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    ddg = DuckDuckGoSearchRun(api_wrapper=DuckDuckGoSearchAPIWrapper())

    tools = [
        Tool(
            name="Wikipedia",
            func=wiki.run,
            description="Use this tool to search Wikipedia for factual and encyclopedic information."
        ),
        Tool(
            name="DuckDuckGo",
            func=ddg.run,
            description="Use this tool to search the web for recent or general information."
        ),
    ]

    # Prompt
    prompt = hub.pull("hwchase17/react")

    # Create agent
    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True,
        handle_parsing_errors=True,
        early_stopping_method="generate"
    )

    # Call agent
    if question:
        response = agent_executor.invoke({"input": question})
        return {"answer": response["output"]}



# New POST endpoint to create an item
class Item(BaseModel):
    name: str
    description: Optional[str] = None

@app.post("/items/")
def create_item(item: Item):
    return {"name": item.name, "description": item.description}
