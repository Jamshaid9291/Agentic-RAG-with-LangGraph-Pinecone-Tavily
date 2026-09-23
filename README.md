# Agentic-RAG-with-LangGraph-Pinecone-Tavily

A stateful **Agentic RAG system** that combines private knowledge retrieval with intelligent web fallback.

Unlike a conventional RAG pipeline that retrieves documents and immediately generates an answer, this system evaluates the quality of retrieved evidence before deciding how to proceed.

The workflow follows a source-aware decision process:

```text
                         ┌──────────────────┐
                         │     Question     │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │  Query Router    │
                         └────────┬─────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
                 KB Query                 Simple Chat
                    │                           │
                    ▼                           ▼
             ┌──────────────┐            ┌──────────────┐
             │ Private KB   │            │ Direct LLM   │
             │ Retrieval    │            │ Response     │
             └──────┬───────┘            └──────────────┘
                    │
                    ▼
             ┌──────────────┐
             │ Evidence     │
             │ Grading      │
             └──────┬───────┘
                    │
             ┌──────┴───────┐
             │              │
          Sufficient       Weak
             │              │
             ▼              ▼
       ┌───────────┐   ┌──────────────┐
       │ Generate  │   │ Tavily Web   │
       │ from KB   │   │ Search       │
       └───────────┘   └──────┬───────┘
                              │
                              ▼
                       ┌──────────────┐
                       │ Web Evidence │
                       │ Grading      │
                       └──────┬───────┘
                              │
                    ┌─────────┴─────────┐
                    │                   │
                 Sufficient           Weak
                    │                   │
                    ▼                   ▼
             ┌──────────────┐    ┌──────────────┐
             │ Generate     │    │ Query        │
             │ from Web     │    │ Rewriting    │
             └──────────────┘    └──────┬───────┘
                                        │
                                        ▼
                                  Retry KB Search
```

## Overview

This project demonstrates a **source-aware Agentic RAG architecture** designed for situations where a private knowledge base may not contain enough information to answer a user's question.

The system prioritizes the private knowledge base first. Retrieved evidence is then evaluated by an LLM-based grader.

If the evidence is sufficient, the response is generated strictly from the private knowledge base.

If the evidence is insufficient, the system automatically falls back to live web search through Tavily.

The workflow also includes:

* Query routing
* Private vector retrieval
* Evidence grading
* Web search fallback
* Query rewriting
* Controlled retries
* Source tracking
* Stateful graph execution
* Grounded answer generation

The entire decision flow is orchestrated using **LangGraph**.

---

# Key Design Principles

### 1. Private Knowledge First

The system does not immediately send every question to the public web.

For knowledge-based questions, the private vector database is queried first.

This creates a source-priority strategy:

```text
Private Knowledge Base
        ↓
Evidence Evaluation
        ↓
Use Private KB if sufficient
        ↓
Otherwise → External Web Search
```

This pattern is particularly useful for applications where internal documentation should be preferred over external information.

---

### 2. Retrieval Is Evaluated Before Generation

Traditional RAG commonly follows:

```text
Question → Retrieve → Generate
```

This project introduces an additional decision layer:

```text
Question
   ↓
Retrieve
   ↓
Grade Evidence
   ↓
Sufficient?
 ┌─┴─┐
Yes  No
 │    │
 ▼    ▼
LLM  Web Search
```

The system therefore does not blindly assume that retrieved chunks are relevant.

---

### 3. Intelligent Routing

A lightweight routing layer determines whether the message requires retrieval or can be answered directly.

```text
User Message
     │
     ▼
  Router
  /    \
 KB    Direct
 │       │
 ▼       ▼
RAG     LLM
```

Examples of retrieval-oriented queries include:

* Agentic RAG architecture
* Retrieval grading
* Query rewriting
* LangGraph RAG workflows
* Retriever behavior
* Web fallback strategies

Simple conversational messages can bypass retrieval completely.

---

# Architecture

The application is implemented as a **stateful LangGraph workflow**.

Each node performs a focused responsibility while the shared graph state carries information between stages.

```text
START
  │
  ▼
Route Question
  │
  ├──────────────► Direct Answer ──────► END
  │
  ▼
Retrieve Private KB
  │
  ▼
Grade KB Evidence
  │
  ├── Good ──────► Generate from KB ──► END
  │
  ▼
Search Web
  │
  ▼
Grade Web Evidence
  │
  ├── Good ──────► Generate from Web ─► END
  │
  ▼
Rewrite Query
  │
  ▼
Retry Retrieval
  │
  └──────────────► Controlled Retry
```

The graph makes the decision process explicit instead of hiding the entire workflow inside a single chain.

---

# Technology Stack

| Component            | Technology                               |
| -------------------- | ---------------------------------------- |
| Orchestration        | LangGraph                                |
| LLM                  | Groq                                     |
| LLM Model            | GPT-OSS 20B                              |
| Embeddings           | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Database      | Pinecone                                 |
| Web Search           | Tavily                                   |
| Document Loading     | LangChain WebBaseLoader                  |
| Text Splitting       | RecursiveCharacterTextSplitter           |
| Structured Decisions | Pydantic + LangChain Structured Output   |
| Language             | Python                                   |

---

# Knowledge Ingestion Pipeline

The private knowledge base is created through a standard document-processing pipeline:

```text
Source Document
      ↓
Document Loader
      ↓
Text Extraction
      ↓
Recursive Chunking
      ↓
Local Embeddings
      ↓
Pinecone
      ↓
Retriever
```

The current demonstration indexes LangGraph documentation as the private knowledge source.

The architecture itself is source-agnostic and can be adapted to:

* Internal documentation
* PDFs
* Technical manuals
* Product documentation
* Company policies
* Knowledge bases
* Research papers
* Support documentation

---

# Document Chunking

The project uses `RecursiveCharacterTextSplitter` with:

```text
Chunk Size   : 1000
Overlap      : 150
```

Chunk overlap helps preserve contextual continuity between neighboring chunks.

The chunking strategy can later be tuned according to document type, retrieval quality, and evaluation results.

---

# Embedding Layer

The project uses the lightweight local embedding model:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The model generates:

```text
384-dimensional embeddings
```

Embeddings are normalized before being stored and queried against the vector database.

This keeps embedding generation independent from the external LLM provider.

---

# Vector Retrieval with Pinecone

Pinecone is used as the vector retrieval layer.

Configuration:

```text
Index:
industry-agentic-rag-kb

Namespace:
langgraph-agentic-rag

Dimension:
384

Metric:
Cosine Similarity

Top-K:
4
```

Namespaces provide logical separation of indexed content and make the architecture easier to extend toward multiple datasets or tenants.

---

# Evidence Grading

One of the central components of the system is the evidence grader.

After retrieving documents, an LLM evaluates whether the retrieved evidence can actually answer the user's question.

The grader produces a constrained decision:

```json
{
  "grade": "good"
}
```

or

```json
{
  "grade": "weak"
}
```

This creates an explicit retrieval-quality checkpoint.

### Why this matters

Vector similarity alone does not guarantee that retrieved content contains the information required to answer a question.

The architecture therefore separates:

```text
Retrieval
```

from:

```text
Evidence Validation
```

This provides a cleaner control mechanism for downstream generation.

---

# Web Fallback with Tavily

When the private knowledge base is judged insufficient, the system activates Tavily web search.

```text
Private KB
    │
    ▼
Evidence Weak
    │
    ▼
Tavily Search
    │
    ▼
Web Evidence
    │
    ▼
Evidence Grading
```

This creates a hybrid retrieval architecture:

```text
Private Retrieval + External Retrieval
```

The web is therefore treated as a fallback source rather than the default source for every query.

---

# Query Rewriting

If web evidence is still insufficient, the system can rewrite the original question into a more retrieval-friendly query.

Example flow:

```text
Original Question
       ↓
Weak Evidence
       ↓
Query Rewriter
       ↓
Improved Search Query
       ↓
Retry Retrieval
```

The rewritten query preserves the original intent while making the information need more explicit.

This is useful for questions where the initial wording does not align well with the vocabulary used inside documents or search indexes.

---

# Controlled Retry Mechanism

Agentic workflows need explicit termination conditions.

Without a retry guard, an agent can potentially continue searching or rewriting indefinitely.

This project implements:

```python
MAX_RETRIES = 1
```

The retry counter is stored directly inside the graph state.

Therefore:

```text
Weak Evidence
      ↓
Rewrite
      ↓
Retry
      ↓
Still Weak
      ↓
Insufficient Evidence
```

The workflow has a deterministic stopping condition.

---

# Stateful Agent Architecture

The workflow uses a typed `AgentState` object to maintain execution state.

The state contains information such as:

```text
question
current_query
kb_docs
web_results
kb_grade
web_grade
answer
source_used
retry_count
```

This allows different graph nodes to communicate through a shared state rather than relying on hidden variables.

It also makes the workflow easier to inspect and debug.

---

# Source-Aware Generation

The final generation step is explicitly conditioned on the source that passed the evidence evaluation.

### Private KB Path

```text
Question
   ↓
Private Retrieval
   ↓
Good Evidence
   ↓
Generate using KB context
```

The generation prompt instructs the model to use only the retrieved private context.

### Web Path

```text
Question
   ↓
Private Retrieval
   ↓
Weak Evidence
   ↓
Tavily
   ↓
Good Web Evidence
   ↓
Generate using web context
```

The response also identifies whether the information came from the private knowledge base or web search.

---

# Source Tracking

The graph maintains a `source_used` field.

Possible states include:

```text
private_kb
web_search
direct
insufficient_evidence
```

This provides basic observability into how an answer was produced.

Example:

```text
Question
   ↓
Router
   ↓
KB Retrieval
   ↓
KB Grader
   ↓
Web Fallback
   ↓
Web Grader
   ↓
Generation

Source Used: web_search
```

This becomes particularly useful when debugging unexpected answers.

---

# Structured LLM Decisions

The router and evidence graders use structured outputs instead of relying on free-form LLM responses.

For example:

```python
class EvidenceGrade(BaseModel):
    grade: Literal["good", "weak"]
```

This constrains the decision space and reduces the need for fragile string parsing.

The router similarly restricts its decision to:

```text
kb
direct
```

This is an important pattern for agentic systems where LLM outputs control workflow execution.

---

# Failure Handling

The system explicitly handles the case where neither the private knowledge base nor web search provides sufficient evidence.

Instead of forcing an answer, it terminates with:

```text
insufficient_evidence
```

The assistant then communicates that reliable evidence was not found.

This follows a safer RAG principle:

```text
Insufficient Evidence
        ↓
Do Not Fabricate
        ↓
Return Controlled Failure
```

---

# Example Execution Paths

## Path 1 — Private Knowledge Base

```text
Question
  ↓
Router
  ↓
KB Retrieval
  ↓
KB Evidence = Good
  ↓
Generate from Private KB
```

Expected source:

```text
private_kb
```

---

## Path 2 — Web Fallback

```text
Question
  ↓
Router
  ↓
KB Retrieval
  ↓
KB Evidence = Weak
  ↓
Tavily Search
  ↓
Web Evidence = Good
  ↓
Generate from Web
```

Expected source:

```text
web_search
```

---

## Path 3 — Direct Conversation

```text
Hello
  ↓
Router
  ↓
Direct Answer
```

Expected source:

```text
direct
```

---

## Path 4 — Retrieval Recovery

```text
Question
  ↓
KB Retrieval
  ↓
Weak Evidence
  ↓
Web Search
  ↓
Weak Evidence
  ↓
Query Rewrite
  ↓
Retry
  ↓
Final Decision
```

The retry mechanism prevents uncontrolled looping.

---

# Why LangGraph?

LangGraph is used because the workflow contains explicit state transitions and conditional routing.

A conventional sequential chain would make the architecture harder to express cleanly.

LangGraph allows the system to model:

* Conditional execution
* Stateful processing
* Retry loops
* Retrieval decisions
* Source-specific generation
* Explicit termination
* Inspectable execution paths

The resulting architecture is closer to an orchestration graph than a linear RAG chain.

---

# Project Structure

```text
Agentic-RAG-with-LangGraph-Pinecone-Tavily/
│
├── Agentic_RAG.ipynb
├── README.md
└── .env
```

> API credentials should never be committed to version control.

Recommended `.gitignore` entry:

```text
.env
.ipynb_checkpoints/
__pycache__/
```

---

# Environment Variables

The application expects:

```text
GROQ_API_KEY
TAVILY_API_KEY
PINECONE_API_KEY
```

These should be supplied through environment variables or a local `.env` file.

---

# Getting Started

## 1. Clone the Repository

```bash
git clone <your-repository-url>
cd Agentic-RAG-with-LangGraph-Pinecone-Tavily
```

## 2. Install Dependencies

Install the packages required by the notebook.

## 3. Configure Environment Variables

Create a local `.env` file:

```env
GROQ_API_KEY=your_groq_key
TAVILY_API_KEY=your_tavily_key
PINECONE_API_KEY=your_pinecone_key
```

## 4. Run the Notebook

Open:

```text
Agentic_RAG.ipynb
```

Run the cells sequentially to:

1. Load the knowledge source
2. Create document chunks
3. Generate embeddings
4. Connect to Pinecone
5. Initialize Groq
6. Initialize Tavily
7. Build the LangGraph workflow
8. Visualize the graph
9. Execute test queries

---

# Evaluation Scenarios

The notebook includes several execution scenarios designed to exercise different workflow paths.

### Private Knowledge Query

Tests whether the system can answer using indexed knowledge.

### External Knowledge Query

Tests the transition from private retrieval to Tavily web search.

### Direct Conversation

Tests retrieval bypass for simple conversational input.

### Current / External Query

Tests the system's ability to use external search when the private knowledge base does not contain sufficient information.

---

# Engineering Patterns Demonstrated

This project focuses on several patterns commonly used when designing retrieval-augmented AI systems:

* **Hybrid retrieval architecture**
* **Conditional routing**
* **Evidence-aware generation**
* **LLM-based retrieval grading**
* **Query rewriting**
* **Bounded retries**
* **Stateful orchestration**
* **Structured LLM outputs**
* **Vector similarity search**
* **Source attribution**
* **Controlled failure states**
* **Private-first information retrieval**
* **External knowledge fallback**

---

# Current Limitations

This implementation is intentionally focused on the orchestration architecture rather than being a complete production platform.

Current areas for improvement include:

### Retrieval Evaluation

The current system uses LLM-based binary evidence grading.

A production implementation could additionally evaluate:

* Recall@K
* Precision@K
* MRR
* NDCG
* Context relevance
* Answer faithfulness
* Answer correctness

### Observability

The workflow currently exposes basic source tracking and console logging.

A more advanced implementation could add:

* LangSmith tracing
* Token/cost monitoring
* Latency monitoring
* Retrieval diagnostics
* Per-node execution metrics
* Failure analytics

### Security

A production deployment would require additional controls around:

* Authentication
* Authorization
* Tenant isolation
* Secret management
* Input validation
* Data access policies
* Prompt injection defenses

### Knowledge Ingestion

The current demonstration uses a single web document as the private knowledge source.

The ingestion pipeline could be extended to support:

```text
PDF
DOCX
HTML
Markdown
Databases
Cloud Storage
Internal APIs
```

---

# Future Architecture

The current design can evolve into a larger multi-source AI system:

```text
                         User Query
                             │
                             ▼
                       Intent Router
                             │
          ┌──────────────────┼──────────────────┐
          ▼                  ▼                  ▼
      Private KB         SQL/Data           Web Search
          │                  │                  │
          └──────────────────┼──────────────────┘
                             ▼
                     Evidence Evaluation
                             │
                             ▼
                       Answer Synthesis
                             │
                             ▼
                    Source Attribution
```

Additional specialized agents could later be introduced for:

* SQL/database reasoning
* Document retrieval
* Web research
* API/tool execution
* Data analysis
* Citation verification

---

# Key Takeaway

The main objective of this project is not simply to implement RAG.

It demonstrates how a retrieval system can be turned into a **decision-driven agentic workflow**.

Instead of assuming:

```text
Retrieve → Generate
```

the system evaluates the information available at each stage:

```text
Route
  ↓
Retrieve
  ↓
Evaluate Evidence
  ↓
Choose Source
  ↓
Search / Rewrite if Required
  ↓
Generate
  ↓
Track Source
```

This architecture provides a practical foundation for building more complex **stateful, tool-using and source-aware AI systems** with LangGraph.
