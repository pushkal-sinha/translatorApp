import os 
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage,SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st

load_dotenv()

st.title("Translator App")
groq_api_key = os.getenv("groq_api_key")

model = ChatGroq(model='Gemma2-9b-It',groq_api_key=groq_api_key)

parse = StrOutputParser()

template = ChatPromptTemplate.from_messages(
    [("system","Translate the following {sourceLanguage} text into {targetLanguage} text : "),("user","{text}")]
)

sourceLanguage = st.text_input('Source Language (Set the source language like : ENGLISH)')
targetLanguage = st.text_input('Target Language (Set the target language like : FRENCH)')
text=st.text_input('Enter text')

chain = template|model|parse
st.text_area('Translated text',value=chain.invoke({"sourceLanguage":sourceLanguage,"targetLanguage":targetLanguage,"text":text}),disabled=True)