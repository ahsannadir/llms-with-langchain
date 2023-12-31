from dotenv import load_dotenv
from langchain.llms import HuggingFaceHub
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import streamlit as st

load_dotenv()

hub_llm = HuggingFaceHub(
    repo_id="openchat/openchat-3.5-1210",
    model_kwargs = {'temperature': 0.9, 'max_length': 1500}
)

prompt = PromptTemplate(
    input_variables = ["topic"],
    template = "Write an essay on the topic: {topic}"
)

hub_chain = LLMChain(prompt = prompt, llm = hub_llm, verbose = True)

st.title("Essay Writer")
user_input = st.text_input("Enter a topic:")
st.write(hub_chain.run(user_input))