import os
import glob
import re 
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
import torch

# Define the device and model for the embeddings
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
model_name = "Qwen/Qwen3-Embedding-0.6B"
print(f"Downloading embeddings from {model_name} on {device}")

# Define the course folder for the vectorstore database
folders = glob.glob("ML") # or SC
db_name = "ml_vstore" # or sc_vstore
embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"device": device})
text_loader_kwargs = {'encoding': 'utf-8'}

documents = []
for folder in folders:
    doc_type = os.path.basename(folder)
    loader = DirectoryLoader(folder, glob="**/*.md", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)
    folder_docs = loader.load()
    
    for doc in folder_docs:
        doc.metadata["doc_type"] = doc_type
        source_path = doc.metadata.get("source", "")
        filename = os.path.basename(source_path)
        match = re.search(r"lecture_(\d+)_slide_(\d+)", filename)
        
        if match:
            doc.metadata["lecture_id"] = int(match.group(1))
            doc.metadata["slide_id"] = int(match.group(2))
        documents.append(doc)

vectorstore = Chroma.from_documents(documents=documents, embedding=embeddings, persist_directory=db_name)
print(f"Vectorstore created with {vectorstore._collection.count()} documents")