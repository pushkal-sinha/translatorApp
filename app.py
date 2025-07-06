from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()

if os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
    os.environ["LANGCHAIN_TRACING_V2"]="true"
    os.environ["LANGCHAIN_PROJECT"]="GROQ Chatbot"

prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are {character} and act like it. Say something famous that {character} says and then answer the queries in brief."),
        ("user","Question:{question}")
    ]
)

def generate_response(question,api_key,engine,character):
    llm=ChatGroq(model=engine,api_key=api_key)
    output_parser=StrOutputParser()
    chain=prompt|llm|output_parser
    if character:
        answer=chain.invoke({'question':question,'character':character})
    else:
        answer=chain.invoke({'question':question,'character':"A helpful chat assisstant"})
    return answer

st.title("Quirky chatbot")

## Sidebar for settings
st.sidebar.title("Settings")
api_key=st.sidebar.text_input("Enter API Key:",type="password")

## Select the model
engine=st.sidebar.selectbox("Select model",["qwen-qwq-32b","gemma2-9b-it","deepseek-r1-distill-llama-70b","llama-3.3-70b-versatile"])
character=st.sidebar.selectbox("Select my character",["Michael Scott from The Office","Sheldon from Big Bang Theory","Phoebe from FRIENDS","Thanos from The Avengers"])


## MAin interface for user input
st.write("Go ahead and ask any question")
user_input=st.text_input("You:")

if user_input and api_key:
    response=generate_response(user_input,api_key,engine,character)
    st.write(response)

elif user_input:
    st.warning("Please enter the OPen AI aPi Key in the sider bar")
else:
    st.write("Please provide the user input")

