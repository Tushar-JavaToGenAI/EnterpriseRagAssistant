import os
import glob
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader

class EnterpriseDocumentLoader:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir

    def load_all(self) -> List[Document]:
        documents = []
        for filepath in glob.glob(f"{self.data_dir}/*.*"):
            filename = os.path.basename(filepath)
            try:
                if filepath.endswith(".pdf"):
                    loader = PyPDFLoader(filepath)
                elif filepath.endswith(".md") or filepath.endswith(".txt"):
                    loader = TextLoader(filepath, encoding="utf-8")
                else:
                    continue

                docs = loader.load()
                for d in docs:
                    d.metadata["source"] = filename
                documents.extend(docs)
                print(f"[Loader] Successfully loaded: {filename}")
            except Exception as e:
                print(f"[Loader] Error loading {filepath}: {e}")
        return documents