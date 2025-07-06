from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage, SystemMessage
import streamlit as st
import os
from dotenv import load_dotenv
from datetime import datetime
load_dotenv()

if 'store' not in st.session_state:
    st.session_state.store = {}
if 'session_id' not in st.session_state:
    st.session_state.session_id = 'Chat_' + datetime.now().strftime("%Y-%m-%d-%H:%M:%S")

config={"configurable":{"session_id":st.session_state.session_id}}

if os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
    os.environ["LANGCHAIN_TRACING_V2"]="true"
    os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGCHAIN_PROJECT_NAME")

if os.getenv("GROQ_KEY"):
    api_key=os.getenv("GROQ_KEY")

prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are {character} and act like it. Say something famous that {character} says and then answer the queries in brief."),
        MessagesPlaceholder(variable_name="messages")
    ]
)

def get_session_history(session_id:str)->BaseChatMessageHistory:
    if session_id not in st.session_state.store:
        st.session_state.store[session_id]=ChatMessageHistory()
    return st.session_state.store[session_id]

def generate_response(question,api_key,engine,character):
    llm=ChatGroq(model=engine,api_key=api_key)
    chain=prompt|llm
    with_message_hist=RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="messages"
    )
    answer=with_message_hist.invoke({'messages':[HumanMessage(content=question)],'character':character},config=config).content
    return answer



st.title("Quirky chatbot")

## Sidebar for settings
st.sidebar.title("Settings")

## Select the model
engine=st.sidebar.selectbox("Select model",["gemma2-9b-it","llama-3.3-70b-versatile"])
character_drop=st.sidebar.selectbox("Select my character",["Michael Scott from The Office","Sheldon from Big Bang Theory","Phoebe from FRIENDS","Thanos from The Avengers"])
character_text=st.sidebar.text_input("Or enter your own character/person",placeholder="Ex: Tony Stark from Iron Man")

if character_text:
    character = character_text
else:
    character = character_drop

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

