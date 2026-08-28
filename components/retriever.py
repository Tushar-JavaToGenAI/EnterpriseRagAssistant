from typing import List
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from sentence_transformers import CrossEncoder

class EnterpriseHybridRetriever:
    def __init__(self, vector_store: Chroma, all_chunks: List[Document], top_k: int = 10):
        # 1. Dense Semantic Retriever
        dense_retriever = vector_store.as_retriever(search_kwargs={"k": top_k})

        # 2. Sparse Lexical BM25 Retriever
        sparse_retriever = BM25Retriever.from_documents(all_chunks)
        sparse_retriever.k = top_k

        # 3. Hybrid Ensemble (50% Dense / 50% Sparse)
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[dense_retriever, sparse_retriever],
            weights=[0.5, 0.5]
        )

        # 4. Cross-Encoder Model for Reranking
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def retrieve_and_rerank(self, query: str, top_n: int = 3) -> List[Document]:
        # Step 1: Hybrid Retrieval candidate pool
        candidates = self.ensemble_retriever.invoke(query)
        if not candidates:
            return []

        # Step 2: Content Deduplication
        unique_docs = list({d.page_content: d for d in candidates}.values())

        # Step 3: Cross-Encoder Scoring
        pairs = [[query, doc.page_content] for doc in unique_docs]
        scores = self.reranker.predict(pairs)

        # Step 4: Sort and pick top N
        scored_docs = sorted(zip(unique_docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, score in scored_docs[:top_n]]