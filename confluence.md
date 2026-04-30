# RAG — Architecture Team Knowledge Base

## Goal

Build a RAG that will use architecture team formal documents such as ARB, GARB, and GIB presentation supports.

**Github repository :** [noe-godin/Architect-s-RAG](https://github.com/noe-godin/Architect-s-RAG)

---

## Key Elements to Build a RAG

| Step | Description |
|------|-------------|
| **Ingestion & Parsing** | Extract text, graph info, and metadata from PDF |
| **Chunking** | Divide file content into smaller chunks instead of giving the whole file to the AI |
| **Embedding** | Transform chunks into vectors |
| **Vector Store** | Store the vectors in a dedicated database |
| **Retrieval** | Transform the user's question into a vector and retrieve relevant chunks from the database |
| **Response** | Generate a structured prompt and send it to the LLM |

---

## Tools & Framework

**Orchestration :** [LangChain](https://python.langchain.com/docs/get_started/introduction) — acts as an orchestrator for the full RAG pipeline.

**Documents storage :** Sharepoint / Blob Storage / Local

---

### Data Extraction

| Name | PyPDF | PyMuPDF | PDFPlumber |
|------|-------|---------|------------|
| Repository | [py-pdf/pypdf](https://github.com/py-pdf/pypdf) | [pymupdf/PyMuPDF](https://github.com/pymupdf/PyMuPDF) | [jsvine/pdfplumber](https://github.com/jsvine/pdfplumber) |
| Stars | ~10k | ~10k | ~10k |
| Pros | Good for text and metadata retrieval | Excels at vector graphics extraction, multiple formats, shortest extraction speed | Works best on machine-generated PDFs |
| Cons | — | — | Longest extraction speed |
| Currently Used | **Yes** | No | No |

> Could also use a multimodal approach with a local AI:
> - Extract everything using a VLM (e.g. llava)
> - Detect images during ingestion, send to VLM, store textual description
> - Use OpenCLIP to transform images and text into vectors so the system can retrieve images from questions
>
> ⚠️ All these options require high GPU output.

---

### Embedding Model

[MTEB Leaderboard — Hugging Face](https://huggingface.co/spaces/mteb/leaderboard)

| Name | nomic-embed-text | BGE-M3 | qwen3-embedding:4b | qwen3-embedding:8b |
|------|-----------------|--------|-------------------|-------------------|
| Link | [nomic-embed-text](https://ollama.com/library/nomic-embed-text) | [bge-m3](https://ollama.com/library/bge-m3) | [qwen3-embedding:4b](https://ollama.com/library/qwen3-embedding) | [qwen3-embedding:8b](https://ollama.com/library/qwen3-embedding) |
| Dimensions | 768 | 1024 | 2560 | 4096 |
| Max tokens | 8192 | 8192 | 32768 | 32768 |
| Multilingual | No (English-focused) | Yes (100+ languages) | Yes | Yes |
| Pros | Lightweight, fast | Multi-functional: dense + sparse + multi-vector retrieval, strong on MTEB, excellent for multilingual docs | Strong multilingual, long context | Strongest quality, long context |
| Cons | English-focused, lower MTEB score | Heavier than nomic | Slower ingestion | Heavy, requires good GPU |
| Speed (3 PDFs) | ~1 min | — | — | — |
| Currently Used | **Yes** | No | No | No |

---

### Vector Database

| Name | Milvus | Chroma | PGVector | Qdrant |
|------|--------|--------|----------|--------|
| Repository | [milvus-io/milvus](https://github.com/milvus-io/milvus) | [chroma-core/chroma](https://github.com/chroma-core/chroma) | [pgvector/pgvector](https://github.com/pgvector/pgvector) | [qdrant/qdrant](https://github.com/qdrant/qdrant) |
| Stars | ~44k | ~28k | ~21k | ~31k |
| Pros | GPU-accelerated, billions of vectors, multiple index types, schema enforcement | Simple Python integration, no server needed, persistent disk storage | Postgres extension — vectors and app data in same table, standard SQL | Written in Rust, rich filtering, supports quantization |
| Cons | Complex to self-host, steep learning curve, overkill under millions of vectors | Degrades above hundreds of thousands of vectors, no horizontal scaling | No built-in horizontal scaling, requires Postgres expertise | Small community |
| Best Scale | Billions | Hundreds of thousands | Millions | Hundreds of millions |
| Best For | Enterprise scale | Prototyping | Already using Postgres | Dedicated vector DB |
| Currently Used | No | **Yes** | No | No |

---

### Chat Model

| Name | llama3.1 | Mistral | Qwen2.5 | SecureGPT |
|------|----------|---------|---------|-----------|
| Size | 8B | 7B | 7B–72B | — |
| Multilingual | Mostly English | Mostly English | Yes (French included) | — |
| Pros | Good general reasoning, runs fully locally | Very fast, lightweight, strong instruction following | Native French/English support, strong reasoning, wide size range | Hosted solution, no local setup |
| Cons | Weaker on non-English content | Less capable on complex reasoning than llama3.1 | Heavier at larger sizes | Data may leave the machine |
| Currently Used | **Yes** | No | No | No |

---

## Our Implementation

### 2-Step Process: Ingestion / Retrieve and Generate

The system works in two distinct phases. The first phase, **Ingestion**, is run once (or whenever new documents are added). The second phase, **Retrieve and Generate**, runs on every user query.

**Ingestion** is handled by `core/ingestion.py`. It loads all PDF files from the `assets/` directory using the configured data extraction library, divides the content into chunks using a recursive character splitter, enriches each chunk with a `source_stem` metadata field derived from the filename (e.g. `ARB_PA_NetReveal`), embeds each chunk using the configured embedding model running via Ollama, and stores the resulting vectors in ChromaDB under the `coArchi` collection. This step only needs to be re-run when new documents are added to `assets/`.

**Retrieve and Generate** is triggered on every user interaction via `app.py`. When the user submits a question, the system first tries to detect a project name in the query by matching tokens against the known `source_stem` values indexed in ChromaDB — if a match is found, the vector search is filtered to only look inside that specific document. It then calls `build_retriever()` which, depending on the `RETRIEVAL_TYPE` config, will either run a standard similarity search or an MMR search to fetch a larger set of candidates. If reranking is enabled, BM25 re-scores those candidates by term frequency and keeps only the top results. The final chunks are assembled into a context string and passed to `ChatOllama` with a structured prompt that instructs the LLM to answer solely from the provided context. The response is streamed token by token to the Streamlit UI.

### Key Config (`config.py`)

| Parameter | Value | Role |
|-----------|-------|------|
| `RETRIEVAL_K` | `2` | Final number of chunks returned after reranking |
| `RETRIEVAL_K_FETCH` | `10` | Candidates fetched from vector DB before reranking |
| `RERANK_ENABLED` | `True` | Toggle BM25 reranking |
| `RETRIEVAL_TYPE` | `SIMILARITY` | `similarity` or `mmr` |
| `chunk_size` | `1000` | Max chunk size in characters |
| `chunk_overlap` | `200` | Overlap between consecutive chunks |
| Collection | `coArchi` | ChromaDB collection name |

---

## Diagnostic of Error / Imprecision

| Layer | Possible Issues |
|-------|----------------|
| **Data** | Missing documents in `assets/`, PDFs not re-ingested after adding new files |
| **Parsing** | Scanned PDFs (images) not readable by PyPDF — need OCR or multimodal approach |
| **Embedding** | Ollama not running, model not pulled (`nomic-embed-text`) |
| **Retrieving** | `source_stem` filter too aggressive — token matched wrong document; try disabling project filter |
| **LLM / Chat** | Ollama not running, `llama3.1` model not pulled; answer hallucinated outside context |
