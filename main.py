from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import os

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model = "gemini-3.5-flash",
    google_api_key = os.getenv("GOOGLE_API_KEY"),
    temperature = 0.3    
)

response = llm.invoke("Explain RAG in two sentences.")

print(response.content)