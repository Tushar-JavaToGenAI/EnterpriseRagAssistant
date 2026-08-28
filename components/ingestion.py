import os
from typing import List
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

class EnterpriseIngestionPipeline:
    def __init__(self, chroma_dir: str = "cache/chroma_db", cache_dir: str = "cache/embedding_cache"):
        self.chroma_dir = chroma_dir
        os.makedirs(self.chroma_dir, exist_ok=True)

        # Runs locally on your machine without external API calls
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

    def ingest(self, chunks: List[Document]) -> Chroma:
        return Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.chroma_dir
        )

    def load_vector_store(self) -> Chroma:
        return Chroma(
            persist_directory=self.chroma_dir,
            embedding_function=self.embeddings
        )