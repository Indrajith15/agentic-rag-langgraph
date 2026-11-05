from langchain_chroma import Chroma
from embeddings import embeddings_model, generate_embeddings

def store_embeddings():

    chunks, vectors = generate_embeddings()

    vector_store = Chroma(
        collection_name="technical_knowledge",
        embedding_function=embeddings_model,
        persist_directory="./chroma_db"
    )

    if vector_store._collection.count() == 0:

        vector_store.add_texts(
            texts=chunks,
            embeddings=vectors,
            ids=[f"chunk_{i}" for i in range(len(chunks))],
            metadatas=[
                {"source": "cloud_computing.txt"}
                for _ in chunks
            ]
        )

    return vector_store

if __name__ == "__main__":
    vector_store = store_embeddings()

    print("Documents successfully stored in ChromaDB!")
    print("Total Documents:", vector_store._collection.count())