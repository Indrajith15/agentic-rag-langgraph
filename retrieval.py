from vector_store import store_embeddings

def retrieve_documents(question):

    vector_store = store_embeddings()

    results = vector_store.similarity_search(
        query = question,
        k=2
    )
    return results


if __name__ == "__main__":
    question = "What AWS service is used to store files?"

    results = retrieve_documents(question)

    for index, document in enumerate(results):
        print(f"\nRESULT {index+1}:")
        print("Content:", document.page_content)
        print("Metadata:", document.metadata)

