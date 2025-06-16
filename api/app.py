from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langserve import add_routes
import uvicorn
import os
from langchain_ollama.llms import OllamaLLM
from dotenv import load_dotenv

load_dotenv()

os.environ['OpenAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")

app=FastAPI(
    title='Langchain Server',
    version='1.0',
    description='A simple API Server',
)
'''
add_routes(
    app,
    ChatOpenAI(),
    path="/openai",
)
'''

llm1=OllamaLLM(model="gemma3")
llm2=OllamaLLM(model="deepseek-r1")

prompt1=ChatPromptTemplate.from_template("Write me an essay about {topic} with 100 words")
prompt2=ChatPromptTemplate.from_template("Write me an poem about {topic} for a 5 years child with 100 words")

add_routes(
    app,
    prompt1|llm1,
    path="/essay"


)

add_routes(
    app,
    prompt2|llm2,
    path="/poem"


)


if __name__=="__main__":
    uvicorn.run(app,host="localhost",port=8000)