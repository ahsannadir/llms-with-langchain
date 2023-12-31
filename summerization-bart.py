from dotenv import load_dotenv
from langchain.llms import HuggingFaceHub
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import streamlit as st

load_dotenv()

hub_llm = HuggingFaceHub(
    repo_id="facebook/bart-large-cnn",
    model_kwargs = {'temperature': 0, 'max_length': 1000}
)

prompt = PromptTemplate(
    input_variables = ["text"],
    template = "Summarize this text: {text}"
)

hub_chain = LLMChain(prompt = prompt, llm = hub_llm, verbose = True)

st.title("Text Summerization")
user_input = st.text_area("Enter any long text", height=400)
st.write("Summary:")
st.write(hub_chain.run(user_input))