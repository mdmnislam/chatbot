from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
load_dotenv()

def pdf_loader(file_path):
    loader = DirectoryLoader(
        file_path, glob="*.pdf", loader_cls=PyPDFLoader
    )
    documents = loader.load()
    
    return documents

def filter_contents(documents):
    contents = []
    for doc in documents:
        content = doc.page_content
        source = doc.metadata.get("source")
        contents.append(
            Document(
                page_content=content,
                metadata={"source": source}
            )
        )
    
    return contents

def text_split(contents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=30
    )
    chunks = text_splitter.split_documents(contents)

    return chunks

def text_embeddings():
    embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')  #this model return 384 dimensions
    
    return embeddings


    
    