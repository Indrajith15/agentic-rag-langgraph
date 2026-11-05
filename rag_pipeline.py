from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from retrieval import retireve_documents
import os

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model = "gemini-3.5-flash",
    google_api_key = os.getenv("Google_API_KEY"),
    temperature = 0.3
)

def rag_pipeline(question):
    documents = retireve_documents(question)

    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    prompt = f"""
    You are a technical knoweldge assistant.
    Answer the user's question using ONLY the provided context.
    If the answer is not available in the context,
    clearly say that you don't have enough information.

    context:
    {context}

    question:
    {question}

    Answer:
    """

    response = llm.invoke(prompt)
    content = response.content
    if isinstance(content, list):
        return "\n".join(
            block["text"]
            for block in content
            if isinstance(block, dict) and "text" in block
        )
    return content

if __name__ == "__main__":
    question = "What is the monthly cost of an EC2 instance?"
    answer = rag_pipeline(question)

    print("\nFinal ANSWER:\n")
    print(answer)