def load_document():
    with open("document/cloud_computing.txt", "r", encoding = "utf-8") as file:
        content = file.read()

    return content

if __name__ == "__main__":
    document = load_document()
    print(document)