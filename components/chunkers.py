import re
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

class FixedRecursiveChunker:
    """Standard sliding window character chunker with overlap."""
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        return self.splitter.split_documents(documents)


class TableAwareChunker:
    """Specialized chunker for table structures that preserves headers."""
    def __init__(self, chunk_size: int = 500):
        self.chunk_size = chunk_size

    def is_table_block(self, block: str) -> bool:
        lines = [l.strip() for l in block.strip().split("\n") if l.strip()]
        if len(lines) >= 2:
            return all(l.startswith("|") and l.endswith("|") for l in lines[:2])
        return False

    def split_table(self, table_text: str, metadata: dict) -> List[Document]:
        lines = [l for l in table_text.strip().split("\n") if l.strip()]
        if len(lines) <= 2:
            return [Document(page_content=table_text, metadata={**metadata, "chunk_type": "table"})]

        header = "\n".join(lines[:2])
        data_rows = lines[2:]
        chunks = []
        current_rows = []

        for row in data_rows:
            current_rows.append(row)
            candidate = header + "\n" + "\n".join(current_rows)
            if len(candidate) > self.chunk_size:
                if len(current_rows) > 1:
                    current_rows.pop()
                    chunks.append(Document(
                        page_content=header + "\n" + "\n".join(current_rows),
                        metadata={**metadata, "chunk_type": "table"}
                    ))
                    current_rows = [row]
                else:
                    chunks.append(Document(
                        page_content=candidate,
                        metadata={**metadata, "chunk_type": "table"}
                    ))
                    current_rows = []

        if current_rows:
            chunks.append(Document(
                page_content=header + "\n" + "\n".join(current_rows),
                metadata={**metadata, "chunk_type": "table"}
            ))
        return chunks


class EnterpriseAdaptiveChunker:
    """
    Routes tabular blocks to TableAwareChunker and narrative text
    to FixedRecursiveChunker with overlap.
    """
    def __init__(self, chunk_size: int = 450, chunk_overlap: int = 50):
        self.fixed_chunker = FixedRecursiveChunker(chunk_size, chunk_overlap)
        self.table_chunker = TableAwareChunker(chunk_size)

    def split_documents(self, documents: List[Document]) -> List[Document]:
        final_chunks = []
        for doc in documents:
            blocks = re.split(r'\n\s*\n', doc.page_content)
            for block in blocks:
                if not block.strip():
                    continue
                if self.table_chunker.is_table_block(block):
                    final_chunks.extend(self.table_chunker.split_table(block, doc.metadata))
                else:
                    sub_doc = Document(page_content=block, metadata={**doc.metadata, "chunk_type": "text"})
                    final_chunks.extend(self.fixed_chunker.split_documents([sub_doc]))
        return final_chunks