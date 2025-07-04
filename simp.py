import os 
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage,SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st

load_dotenv()

st.title("Simping!")
groq_api_key = os.getenv("groq_api_key")

model = ChatGroq(model='Gemma2-9b-It',groq_api_key=groq_api_key)

parse = StrOutputParser()

instruction1="Give answer to my question assuming I am a hopeless romantic in the language : {targetLanguage}."
instruction2="Assume yourself as a hopeless romantic and generate a response in {targetLanguage} on above conditon "

template1 = ChatPromptTemplate.from_messages(
    [("system",instruction1),("user","{text}")]
)

template2 = ChatPromptTemplate.from_messages(
    [("system",instruction2),("user","{text}")]
)

targetLanguage = st.text_input('Select your Language')
text=st.text_input('Enter your question')

chain = template1|model|parse
chain2= template2|model|parse
response1=chain.invoke({"targetLanguage":targetLanguage,"text":text})
st.text_area('Translated text',value=response1,disabled=True)
st.subheader("A message you could send!")
response2=chain2.invoke({"targetLanguage":targetLanguage,"text":response1})
st.text_area('Translated text',value=response2,disabled=True)