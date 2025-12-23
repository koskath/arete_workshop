import os
import glob
import re 
from langchain_chroma import Chroma
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from dotenv import load_dotenv

# Load environment variables (including MISTRAL_API_KEY)
load_dotenv(override=True)

# Define the model for the embeddings
model_name = "mistral-embed"
print(f"Using Mistral embeddings API with model: {model_name}")

# Define the course folder for the vectorstore database
folders = glob.glob("ML") # or SC
db_name = "ml_vstore" # or sc_vstore
embeddings = MistralAIEmbeddings(model=model_name)
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