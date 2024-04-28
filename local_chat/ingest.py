from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
import os
from constants import CHROMA_SETTINGS

persist_directory = "db"

def main():
    loaders = []

    # Add PDF loader
    for root, dirs, files in os.walk("docs"):
        for file in files:
            if file.endswith(".pdf"):
                print(file)
                loaders.append(PyPDFLoader(os.path.join(root, file)))

    # Add TextFileLoader for .txt files
    for root, dirs, files in os.walk("docs"):
        for file in files:
            if file.endswith(".txt"):
                print(file)
                loaders.append(TextLoader(os.path.join(root, file)))

    # Load documents from all loaders
    documents = []
    for loader in loaders:
        documents.extend(loader.load())

    print("splitting into chunks")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)

    print("Loading sentence transformers model")
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    print(f"Creating embeddings. May take some minutes...")
    db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory, client_settings=CHROMA_SETTINGS)
    db.persist()
    db = None

    print(f"Ingestion complete! You can now run main file")

if __name__ == "__main__":
    main()
