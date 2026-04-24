Goal
Build a RAG that will use architecture's team formal documents such as ARB, GARB, GIB's presentation support.
Github repository : noe-godin/Architect-s-RAG
Implementations
Key elements to build a RAG : 
Ingestion & Parsing : Extract text but also graphs infos and metadata from pdf
Chunking : Instead of giving whole file to AI, need to divide content into chunks
Embedding : Transform these chunks into a vector
Vector Store : Store the Vectors in a special database
Retrieval : The user's question need to be transformed into a vector, and database need to retrieve correct files
Response : Generate a structured prompt to the LLM

ToolsFramework :
Langchain : Acts as an orchestrator Build a RAG agent with LangChain - Docs by LangChain
Documents storage : 
Sharepoint or Blob Storage or Local
Data Extraction : NamePyPDFPyMuPDFPDFPlumberRepositoryGitHub - py-pdf/pypdfGitHub - pymupdf/PyMuPDFGitHub - jsvine/pdfplumberStars10k10k10kPros
good for retrieving text and metadatas
excels at vector graphics extraction
shortest extraction speed
multiple document format support
Works best on machine-generated, rather than scanned
longest extraction speedCurrently usedYes

Could also use a multimodal with local AI : 
Do all the extraction using a VLM (llava)
Detect Image during Ingestion, send image to local VLM model and store textual description
Use OpenCLIP to transform images and text into vectors, so system can retrieve image when asking questions
All these options require high GPU output
Embedding model : 
This link compare the differents Embedding model :
MTEB Leaderboard - a Hugging Face Space by mteb
My result testing these : 
Namenomic-embed-textQwen3-Embedding-4BQwen3-Embedding-4BLinknomic-embed-textqwen3-embedding:4bqwen3-embedding:8bDownloads66m1.7m1.8mPros
lightweight
fast

Cons


Speed for ingesting 3 pdf1 min

Vector database (Best known & Langchain compatible) : NameMilvusChromaPGVectorQdrantRepositoryGitHub - milvus-io/milvusGitHub - chroma-core/chromaGitHub - pgvector/pgvectorGitHub - qdrant/qdrantStars44k28k21k31kPros
GPU-accelerated indexing and search
Support billions of vectors across distributed clusters
schema enforcement with typed fields
Multiple index types
Simple integration
Directly in python, no server to run
Persistent storage to disk
Postgres extension
vectors and application data in same table
standard SQL for queries, filtering, joins
Big ecosystem and community
Written in rust
Rich payload filtering
Support quantizationCons
Complex to self-host
Steep learning curve
Overkill for most workloads under millions of vectors

Performance degrade above hundreds of thousands of vectors
No horizontal scaling
No built-in horizontal scaling for vector indexes
Performance turning at scale requires Postgres expertise
Small communityBest scaleBillionsHundreds of thousandsMillionsHundreds of millionsBest forEnterprise ScalePrototypingAlready using PostgresDedicated vector databaseCurrently Used
Yes

Chat model :Namellama3.1
models SecureGPTLink


Pros


Cons


Currently UsedYes

Diagnostic of error / imprecision

Data : 
Parsing : 
Embedding : 
Retrieving :
LLMChat : 