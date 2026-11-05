from langchain_text_splitters import RecursiveCharacterTextSplitter
from ingestion import load_document

def split_document():
    document = load_document()

    text_splitters = RecursiveCharacterTextSplitter(
        chunk_size = 300,
        chunk_overlap = 50
    )

    chunks = text_splitters.split_text(document)

    return chunks

if __name__ == "__main__":
    chunks = split_document()

    for index, chunk in enumerate(chunks):
        print(f"\nCHUNK {index + 1}:\n")
        print(chunk)

