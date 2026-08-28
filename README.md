Markdown
# Enterprise Knowledge Assistant (Advanced RAG Architecture)

A production-grade, conversational Retrieval-Augmented Generation (RAG) assistant designed for strictly grounded document search and tabular policy extraction across enterprise knowledge repositories.

---

## 🏗️ High-Level System Architecture

                              DATA INGESTION LAYER
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                         │
│   [ Policy Documents ] ──► [ EnterpriseDocumentLoader ]                                 │
│     (.pdf, .txt, .md)         • PyPDFLoader                                             │
│                               • UTF-8 TextLoader                                        │
│                                       │                                                 │
│                                       ▼                                                 │
│                          [ Adaptive Chunker Strategy ]                                  │
│                                       │                                                 │
│                 ┌─────────────────────┴─────────────────────┐                           │
│                 ▼                                           ▼                           │
│     [ FixedRecursive Chunker ]                  [ TableAware Chunker ]                  │
│     • Paragraphs / Narratives                   • Markdown Tables                       │
│     • Chunk: 450 | Overlap: 50                  • Preserves Headers across Chunks       │
│                 │                                           │                           │
│                 └─────────────────────┬─────────────────────┘                           │
│                                       │                                                 │
│                 ┌─────────────────────┴─────────────────────┐                           │
│                 ▼                                           ▼                           │
│      [ Google GenAI Embeddings ]                [ Sparse Lexical Index ]                │
│       (models/text-embedding-004)                   (rank_bm25 Index)                   │
│                 │                                           │                           │
│                 ▼                                           │                           │
│        [( Persistent ChromaDB )]                            │                           │
└─────────────────┬───────────────────────────────────────────┼───────────────────────────┘
│                                           │
│           HYBRID RETRIEVAL & RERANKING    │
┌─────────────────┼───────────────────────────────────────────┼───────────────────────────┐
│                 ▼                                           ▼                           │
│       [ Dense Vector Search ]                     [ Sparse BM25 Search ]                │
│       (Semantic / Conceptual)                     (Exact Match / IDs / Keywords)        │
│                 │                                           │                           │
│                 └─────────────────────┬─────────────────────┘                           │
│                                       ▼                                                 │
│                         [ LangChain EnsembleRetriever ]                                 │
│                          • 50% Dense / 50% Sparse Pool                                  │
│                                       │                                                 │
│                                       ▼                                                 │
│                        [ Content Deduplication Filter ]                                 │
│                                       │                                                 │
│                                       ▼                                                 │
│                         [ Cross-Encoder Re-Ranker ]                                     │
│                         (ms-marco-MiniLM-L-6-v2)                                        │
│                          • Computes deep query-doc cross-attention                      │
│                          • Sorts & slices Top-N most relevant chunks                    │
└───────────────────────────────────────┬─────────────────────────────────────────────────┘
│
│ TOP-N GROUNDED CHUNKS
▼
GENERATION & GUARDRAIL LAYER
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                         │
│   [ User Input Query ] ──► [ Conversational Contextualizer ]                            │
│   + Chat History               • Resolves coreferences into a standalone query          │
│                                       │                                                 │
│                                       ▼                                                 │
│                           [ Strict Grounding Prompt ]                                   │
│                            • Enforces zero-hallucination constraint                     │
│                            • Injects retrieved context + source tags                    │
│                                       │                                                 │
│                                       ▼                                                 │
│                         [ ChatGoogleGenerativeAI Engine ]                               │
│                           (Gemini 1.5 Flash / 2.0 Flash)                                │
│                                       │                                                 │
│                                       ▼                                                 │
│                             [ Streamlit Chat UI ]                                       │
│                            • Formatted grounded answer                                  │
│                            • Collapsible source context & citations                     │
└─────────────────────────────────────────────────────────────────────────────────────────┘


---

## ⚙️ Core Architectural Modules

**1. Data Ingestion & Adaptive Chunking Layer**
* **Multi-Format Ingestion:** Loads unstructured text, markdown files, and PDFs directly into normalized document streams with uniform metadata attribution.
* **Table-Aware Chunker:** Automatically identifies Markdown table syntax and duplicates column headers across segmented row slices, preventing schema fragmentation in tabular HR policies.
* **Recursive Narrative Chunker:** Handles general prose with sliding-window chunk overlaps to retain contextual continuity across split boundaries.

**2. Dual-Engine Hybrid Retrieval Layer**
* **Dense Semantic Branch:** Generates document vectors using Google Gemini Embeddings (`text-embedding-004`) stored persistently in a local ChromaDB instance.
* **Sparse Lexical Branch:** Builds an in-memory BM25 index over all document segments for exact keyword, threshold, and allowance identification.
* **Ensemble Blending:** Aggregates dense and sparse result streams at equal weights (0.5 / 0.5) to capture both semantic intent and exact phrase matches.

**3. Deep Cross-Encoder Re-Ranking**
* Passes the candidate pool from the ensemble retriever through a cross-encoder model (`ms-marco-MiniLM-L-6-v2`).
* Calculates full cross-attention between the query and candidate contents simultaneously, eliminating low-relevance noise and selecting the top $N$ chunks for context injection.

**4. Contextualized & Grounded Generation Layer**
* **Query Contextualization:** Evaluates multi-turn chat history to translate ambiguous follow-ups (e.g., *"Does this apply to part-time staff as well?"*) into self-contained search vectors.
* **Strict Grounding Guardrail:** Prompts the generation engine with a zero-extrapolation directive. If the target information is absent from the retrieved chunks, the system strictly outputs a refusal response rather than fabricating policy facts.
* **Interactive UI & Traceability:** Built with Streamlit, providing real-time chat execution and collapsible source citation drawers that expose the underlying document chunks and metadata.