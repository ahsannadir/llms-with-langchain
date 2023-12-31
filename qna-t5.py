from dotenv import load_dotenv
from langchain.llms import HuggingFaceHub
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import streamlit as st

load_dotenv()

hub_llm = HuggingFaceHub(
    repo_id="google/flan-t5-base",
    model_kwargs = {'temperature': 0.5, 'max_length': 100}
)

prompt = PromptTemplate(
    input_variables = ["question"],
    template = "Please answer to the following question. {question}"
)

hub_chain = LLMChain(prompt = prompt, llm = hub_llm, verbose = True)

st.title("Q/A Bot")
user_input = st.text_input("Enter your Question")
st.write(hub_chain.run(user_input))