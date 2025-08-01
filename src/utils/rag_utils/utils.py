import sys
import os
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[3] 
sys.path.insert(0, str(PROJECT_ROOT))  
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.text import TextLoader
from langchain_core.documents import Document
from src.models.LLM.embeddings import LLMEmbedding
from typing import List

def doc_loader(file_path:str) -> List[Document]:
    if file_path.lower().endswith(".txt"):
        loader = TextLoader(file_path)
        raw_docs = loader.load()                
    elif file_path.lower().endswith(".pdf"):
        loader = PyPDFLoader(file_path)
        raw_docs = loader.load()              
    else:
        raise ValueError(f"Unsupported file type: {file_path}")
    embedding_client = LLMEmbedding().client     

    chunker = SemanticChunker(
        embeddings=embedding_client,
        min_chunk_size=500,                                 
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95                     
    )
    chunked_docs = chunker.split_documents(raw_docs)
    return chunked_docs