from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from src.models.model import llm_model

llm = llm_model
prompt = hub.pull("rlm/rag-prompt")
generation_chain = prompt | llm | StrOutputParser()
