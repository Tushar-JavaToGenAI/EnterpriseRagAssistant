import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from components.loaders import EnterpriseDocumentLoader
from components.chunkers import EnterpriseAdaptiveChunker
from components.ingestion import EnterpriseIngestionPipeline
from components.retriever import EnterpriseHybridRetriever
from components.generator import GroundedAnswerGenerator

from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="Enterprise Knowledge Assistant", page_icon="🏢", layout="wide")

@st.cache_resource
def build_rag_system():
    # 1. Load
    loader = EnterpriseDocumentLoader(data_dir="data")
    raw_docs = loader.load_all()
    
    # 2. Chunk (Table-Aware + Fixed Recursive with Overlap)
    chunker = EnterpriseAdaptiveChunker(chunk_size=450, chunk_overlap=50)
    all_chunks = chunker.split_documents(raw_docs)

    # 3. Ingest with Embedding Cache
    ingestor = EnterpriseIngestionPipeline(
        chroma_dir="cache/chroma_db",
        cache_dir="cache/embedding_cache"
    )
    vector_store = ingestor.ingest(all_chunks)

    # 4. Retriever & Generator
    retriever = EnterpriseHybridRetriever(vector_store, all_chunks)
    generator = GroundedAnswerGenerator()
    return retriever, generator

try:
    retriever, generator = build_rag_system()
except Exception as e:
    st.error(f"Initialization failure: {e}")
    st.stop()

# Session State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

with st.sidebar:
    st.title("⚙️ RAG Architecture")
    st.markdown("""
    * **Loaders:** Multi-format (`PDF`, `MD`, `TXT`)
    * **Chunker:** Adaptive (Fixed-overlap + Table-Aware)
    * **Cache:** `LocalFileStore` Cached Embeddings
    * **Retrieval:** Hybrid (`Dense Chroma + BM25`)
    * **Reranker:** `CrossEncoder MiniLM-L-6`
    * **Generator:** Conversational + Strict Grounding
    """)
    if st.button("🗑️ Reset Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

st.title("🏢 Enterprise Knowledge Assistant")
st.caption("Grounded enterprise search for leave policies, benefits tables, and HR rules.")

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("📚 Source Citations & Retrieved Context"):
                st.markdown("**Cited Sources:** " + ", ".join([f"`{s}`" for s in message["sources"]]))
                for i, ctx in enumerate(message.get("context", []), 1):
                    st.text(f"Chunk {i}:\n{ctx}")

# User Query Processing
if prompt := st.chat_input("Ask a policy question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving, reranking, and generating..."):
            # 1. Rewrite query using conversational history
            standalone_query = generator.contextualize_query(prompt, st.session_state.chat_history)

            # 2. Hybrid Retrieval + Reranking
            top_docs = retriever.retrieve_and_rerank(standalone_query, top_n=3)

            # 3. Grounded Generation
            result = generator.generate_answer(prompt, top_docs, st.session_state.chat_history)

            st.markdown(result["answer"])
            if result["sources"]:
                with st.expander("📚 Source Citations & Retrieved Context"):
                    st.markdown("**Cited Sources:** " + ", ".join([f"`{s}`" for s in result["sources"]]))
                    for i, c in enumerate(result["retrieved_context"], 1):
                        st.text(f"Chunk {i}:\n{c}")

            # Store states
            st.session_state.messages.append({
                "role": "assistant",
                "content": result["answer"],
                "sources": result["sources"],
                "context": result["retrieved_context"]
            })
            st.session_state.chat_history.append(HumanMessage(content=prompt))
            st.session_state.chat_history.append(AIMessage(content=result["answer"]))