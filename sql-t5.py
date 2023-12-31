from dotenv import load_dotenv
from langchain.llms import HuggingFaceHub
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import streamlit as st

load_dotenv()

hub_llm = HuggingFaceHub(repo_id="mrm8488/t5-base-finetuned-wikiSQL")

prompt = PromptTemplate(
    input_variables = ["query"],
    template = "Translate English to SQL: {query}"
)

hub_chain = LLMChain(prompt = prompt, llm = hub_llm, verbose = True)

st.title("English to SQL")
user_input = st.text_input("Enter your query")
st.write(hub_chain.run(user_input))