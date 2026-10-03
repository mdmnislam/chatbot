import os
from dotenv import load_dotenv
from src.helper import pdf_loader, filter_contents, text_split, text_embeddings
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
load_dotenv()

"""
   This file will run once to create a vectorstore from the documents in the 
   data folder. It will create a new index in Pinecone and store the embeddings 
   of the documents in that index.
   
"""

pc_api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=pc_api_key)

file_path = '../data/'
documents=pdf_loader(file_path)
contents = filter_contents(documents)
chunks = text_split(contents)

embeddings = text_embeddings()

index_name = "medical-chatbot"
if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(index_name)

print(f"Index '{index_name}' created successfully.")

vectorstore = PineconeVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    index_name=index_name
)

print(f"Vectorstore created successfully with {len(chunks)} documents.")