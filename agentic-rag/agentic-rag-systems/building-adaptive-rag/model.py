
import os
from dotenv import load_dotenv
# from langchain_openai import ChatOpenAI
# from langchain_aws import ChatBedrock
from langchain_google_genai import ChatGoogleGenerativeAI  


# from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

### chat model
# llm_model = ChatOpenAI(temperature=0)
# llm_model =  ChatBedrock(model_id="us.anthropic.claude-sonnet-4-20250514-v1:0", region_name="us-west-2", temperature=0)   

llm_model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",                     # Latest Gemini model for optimal performance
    temperature=0,                                # Deterministic output for consistent agent behavior
)

### embedding model
embed_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")