import os
import chromadb
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_community.llms import Ollama

def concatenate_content_with_metadata(doc):
    metadata_str = ' | '.join(f"{key}: {value}" for key, value in doc.metadata.items())
    return f"{metadata_str} | {doc.page_content}"

def setup_vector_store(directory_path):
    # Define your headers to split on
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3")
    ]

    # Initialize the text splitter
    text_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

    # Define your embedding function
    embed_model = FastEmbedEmbeddings()

    # Initialize Chroma DB Persistent Client
    client = chromadb.PersistentClient(
       path="/home/project/chromadb5",  # Adjust this path as per your setup
       settings=Settings(),  # Use appropriate settings from chromadb.config
       tenant=DEFAULT_TENANT,
       database=DEFAULT_DATABASE,
    )

    # Create or access a collection
    collection = client.get_or_create_collection("rag5")

    # Initialize an empty list to store documents and ids
    all_docs = []
    all_ids = []

    # Iterate through each markdown file in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith('.md'):
            file_path = os.path.join(directory_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                # Read the content of the file
                content = file.read()

            # Split the content into documents
            docs = text_splitter.split_text(content)

            # Add metadata and generate unique IDs for each document
            for doc_index, doc in enumerate(docs):
                doc.metadata["source"] = filename
                all_docs.append(doc)
                all_ids.append(f"{filename}_{doc_index}")
# Concatenate metadata with page content for each document
    concatenated_documents = [concatenate_content_with_metadata(doc) for doc in all_docs]

    # Initialize Chroma vector store with specific parameters
    vectorstore = Chroma.from_documents(
        documents=all_docs,
        embedding=embed_model,
        persist_directory="/home/project/chromadb5",
        collection_name="rag5"
    )

    # Insert documents into the collection
    collection.add(
        ids=all_ids,
        documents=concatenated_documents,
        metadatas=[doc.metadata for doc in all_docs],  # If you still want to include metadata separately
    )

    # Save the vectorstore and embedding model
    return vectorstore, embed_model
if __name__ == "__main__":
    # Define your directory path containing markdown files
    directory_path = '/home/project/project markdown'
    # Setup vector store and embeddings
    vectorstore, embed_model = setup_vector_store(directory_path)
