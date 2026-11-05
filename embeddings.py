from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from chunking import split_document
import os 

load_dotenv()

embeddings_model = GoogleGenerativeAIEmbeddings(
    model = "gemini-embedding-001",
    google_api_key = os.getenv("Google_API_KEY")
)

def generate_embeddings():
    chunks = split_document()
    vectors = embeddings_model.embed_documents(chunks)

    return chunks, vectors

if __name__ == "__main__":
    chunks, vectors = generate_embeddings()
    print("Total Chunks:", len(chunks))
    print("Total Embedding:", len(vectors))
    print("Vector dimensions:", len(vectors[0]))