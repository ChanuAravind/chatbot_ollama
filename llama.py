from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import streamlit as st
import os
from dotenv import load_dotenv

#assign the environmental variables
load_dotenv()
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
#Langsmith tracking
os.environ['LANGCHAIN_TRACING_V2']="true"

#Prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system","you are an helpful assistant who responds to the user queries. Please be honest with the answers and mention if you are not sure about the answer"),
        ("user","Question:{question}")
    ]
)

#Streamlit 
st.title('Langchain with Ollama')
input_text = st.text_input("Ask a Question")

#Ollama LLM
llm = Ollama(model="llama2")
outputparser=StrOutputParser()
chain=prompt|llm|outputparser

if input_text:
    st.write(chain.invoke({'question':input_text}))