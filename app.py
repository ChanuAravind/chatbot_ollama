from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import streamlit as st
import os
from dotenv import load_dotenv

#Load environmental variables file
load_dotenv()

#assign the environmental variables
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
#Langsmith tracking
os.environ['LANGCHAIN_TRACING_V2']="true"

#Prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system","you are an helpful assistant.Please respond to the user queries"),
        ("user","Question:{question}")
    ]
)

#Streamlit 
st.title('Langchain with OPENAI API')
input_text = st.text_input("Ask a Question")

#openAI LLM
llm = ChatOpenAI(model="gpt-3.5-turbo")
outputparser=StrOutputParser()
chain=prompt|llm|outputparser

if input_text:
    st.write(chain.invoke({'question':input_text}))