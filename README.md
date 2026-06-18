# RAG-based Document Q&A System with Analytics Dashboard

A Retrieval-Augmented Generation (RAG) system that answers questions about the **AWS Customer Agreement**. It exposes a FastAPI backend with SQL-based usage logging and a Streamlit frontend for interacting with the system and viewing usage analytics.

The system parses and chunks the provided PDF, embeds the chunks locally, retrieves the most relevant passages for a query, and uses an LLM to generate a grounded answer   returning the answer together with the source snippets it relied on. Every query is logged to a SQLite database so usage analytics can be computed.

---

## 1. Architecture Overview

```
                ┌──────────────────────────┐
                │   Streamlit Frontend      │
                │  (chat UI + analytics)    │
                └────────────┬─────────────┘
                             │ HTTP (requests)
                             ▼
                ┌──────────────────────────┐
                │      FastAPI Backend      │
                │  /ingest  /ask  /analytics│
                └───┬───────────┬──────────┘
                    │           │
        ┌───────────▼──┐   ┌────▼──────────┐
        │  RAG Pipeline │   │  SQLite DB     │
        │  (rag.py)     │   │ (usage logs)   │
        └───┬───────────┘   └───────────────┘
            │
   ┌────────┼─────────────────────────────┐
   │        │                             │
   ▼        ▼                             ▼
 PyMuPDF   sentence-transformers      FAISS index
 (parse)   all-MiniLM-L6-v2          IndexFlatIP
           (embeddings)              (vector store)
                                          │
                                          ▼
                                    Groq LLM (generation)
```

**Flow of a question (`/ask`):**

1. The user's query is embedded with the same model used for the chunks.
2. FAISS returns the top-k most similar chunks (cosine similarity).
3. If the best score is below a threshold, the query is treated as out-of-scope (no LLM call).
4. Otherwise the retrieved chunks are inserted into a prompt and sent to the LLM.
5. The LLM returns a grounded answer (or a fixed "not found" phrase).
6. The interaction (query, answer, no-answer flag, latency, top score) is logged to SQLite.
7. The answer and its source snippets are returned to the frontend.

**Two separate stores:**
- **FAISS** holds the *document* (chunk embeddings) used for retrieval.
- **SQLite** holds the *usage logs*   used for analytics. It does **not** store document content.

---

## 2. Tech Stack & Key Design Decisions

| Component | Choice | Why |
|---|---|---|
| PDF parsing | **PyMuPDF (`fitz`)** | Fast, reliable text extraction; good with the legal document's layout. |
| Embeddings | **`sentence-transformers/all-MiniLM-L6-v2`** | Compact (384-dim), fast, runs locally with no API key, sufficient for a single document. |
| Vector store | **FAISS `IndexFlatIP`** | Exact brute-force search   at a few hundred chunks, approximate indexes are unnecessary. Inner product on normalized vectors = cosine similarity. |
| LLM (generation) | **Groq   `llama-3.3-70b-versatile`** | Free tier, very fast inference (clean latency numbers), strong instruction-following for the no-hallucination requirement. |
| Backend | **FastAPI** | Pydantic validation, automatic JSON serialization, auto-generated docs. |
| Logging DB | **SQLite** | Zero-config, single-file, sufficient for usage analytics. |
| Frontend | **Streamlit** | Runs as a separate process, calls the API over HTTP. |

### Chunking strategy

- **Chunk size:** a sliding window over words, sized by the model's own tokenizer. all-MiniLM-L6-v2 has a hard limit of 256 tokens anything beyond is silently truncated so I target ~200 tokens to stay safely under that cap with headroom. 200 tokens is large enough to hold a coherent clause with its surrounding context, yet small enough to keep the chunk focused so its embedding doesn't become a diluted average of too many distinct topics.
- **Overlap:** ~15% of the window, so a clause split across a chunk boundary survives intact in at least one chunk. I avoid large overlaps (e.g. 50%) because heavy duplication bloats the index and pushes adjacent chunks close together in vector space, causing top-k retrieval to return near-identical passages instead of diverse ones. I keep it above ~10% because a thinner margin is often shorter than the clauses it needs to protect, so boundary-spanning ideas would still get cut. ~15% sits in the standard 10–20% band enough boundary coverage with minimal redundancy.
- **Token-aware sizing:** each candidate chunk is measured with the model's own tokenizer (`model.tokenizer.encode`). If it exceeds the token limit it is shrunk and re-checked. This keeps every chunk safely **under all-MiniLM-L6-v2's 256-token limit**, avoiding silent truncation (the model discards anything beyond 256 tokens without warning).
- **Why not aggressive cleaning:** transformer embedding models are trained on natural text and rely on punctuation/casing for meaning   especially important for a legal document where section numbers (e.g. `9(b)`) and defined terms are load-bearing. Cleaning is limited to fixing extraction artifacts (non-breaking spaces, line-wrap hyphens, repeated whitespace).

### Retrieval

- **top-k = 4** by default   enough context for the LLM to answer without bloating the prompt.
- Embeddings are **L2-normalized**, so FAISS inner product returns cosine similarity. The query is normalized identically at search time.

### No-hallucination handling (two layers)

1. **Score threshold:** if the best retrieved chunk scores below the threshold, the query is treated as out-of-scope and the LLM is skipped entirely.
2. **Prompt fallback:** the system prompt instructs the model to reply with a fixed phrase ("I could not find the answer in the document.") when the context doesn't contain the answer. The response is checked for this phrase to set the `no_answer` flag.

Generation uses **`temperature=0.1`** to keep answers deterministic and grounded rather than creative.

---

## 3. Project Structure

```
Rag_document_qa/
├── app/
│   ├── main.py          # FastAPI app + the 3 endpoints
│   ├── rag.py           # RAG pipeline: parse, chunk, embed, retrieve, generate
│   ├── database.py      # SQLite: init, log_query, get_analytics
│   ├── models.py        # Pydantic request/response models
│   └── config.py        # constants: chunk size, top-k, model names, paths
├── frontend/
│   └── streamlit_app.py # Streamlit UI (chat + analytics)
├── data/
│   └── AWS_Customer_Agreement.pdf
├── storage/             # generated artifacts (gitignored): index, chunks, db
├── .env         
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 4. Setup & Run Instructions

These steps work end-to-end from a fresh clone.

### Prerequisites

- Python 3.10+
- A free Groq API key (sign up at <https://console.groq.com>)

### Step 1   Clone and create a virtual environment

```bash
git clone https://github.com/faheem-bhat2025/Rag_document_qa.git
cd Rag_document_qa

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
```

### Step 2   Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3   Add your API key

Edit the .env.example file to .env and add your Groq key:

```
GROQ_API_KEY=your_groq_api_key_here
```

### Step 4   Start the backend (FastAPI)

From the `app/` directory:

```bash
cd app
uvicorn main:app --reload --port 8000
```

The interactive API docs are available at <http://localhost:8000/docs>.

### Step 5   Ingest the document

Build the vector index from the PDF (run once). Either click **Execute** on `POST /ingest` in the docs UI, or:

```bash
curl -X POST http://localhost:8000/ingest
```

This parses, chunks, embeds, and indexes the PDF, then saves the index and chunks to `storage/`. On subsequent server restarts the index is loaded from disk automatically, no re-ingestion needed.

### Step 6   Start the frontend (Streamlit)

In a **separate terminal** (with the venv activated), run the frontend as its own process. It calls the FastAPI backend over HTTP:

```bash
cd frontend
streamlit run streamlit_app.py
```

The UI opens at <http://localhost:8501>. Make sure the FastAPI server (Step 4) is running first.

> **Note:** The backend and frontend are two separate processes. Start the FastAPI server first, ingest the document, then launch Streamlit.

---

## 5. API Reference

### `POST /ingest`
Parses, chunks, embeds the provided PDF, builds the FAISS index, and persists it to disk.

**Response:**
```json
{ "status": "ingested", "chunks_indexed": 142 }
```

### `POST /ask`
Runs the RAG pipeline for a query, logs the interaction, and returns the answer with its sources.

**Request:**
```json
{ "query": "What is the late payment interest rate?", "k": 4 }
```

**Response:**
```json
{
  "answer": "AWS may charge interest at 1.5% per month ...",
  "sources": [
    { "text": "...3.1 Service Fees. We calculate and bill...", "score": 0.71 }
  ],
  "no_answer": false,
  "latency_ms": 480.2
}
```

**Edge cases handled:**
- Empty query → `400 Bad Request`
- No document ingested yet → `409 Conflict`
- Malformed request body → `422` (automatic via Pydantic)

### `GET /analytics`
Runs the SQL analytics queries and returns the results as JSON.

**Response:**
```json
{
  "total_queries": 32,
  "most_frequent_questions": [
    { "query": "What is the late payment interest rate?", "times_asked": 4 }
  ],
  "no_answer_queries": [
    { "query": "What is the capital of France?", "created_at": "2026-06-18 10:21:03" }
  ],
  "average_latency_ms": 512.7
}
```

---

## 6. SQL Logging Schema

Every call to `/ask` inserts one row into the `query_logs` table.

```sql
CREATE TABLE IF NOT EXISTS query_logs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    query       TEXT    NOT NULL,
    answer      TEXT,
    no_answer   INTEGER NOT NULL DEFAULT 0,   -- 0 = answered, 1 = not found
    top_score   REAL,                          -- best FAISS similarity score
    latency_ms  REAL    NOT NULL,
    created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);
```

**Column rationale:**
- `query`, `no_answer`, `latency_ms`   required by the three analytics metrics.
- `answer`, `top_score`, `created_at`, `id`   logged for debugging, retrieval-quality analysis, and ordering.

All inserts use **parameterized queries** (`?` placeholders) to prevent SQL injection and handle escaping.

---

## 7. Analytics

The `/analytics` endpoint exposes the following, each backed by a SQL query:

| Metric | SQL |
|---|---|
| Most frequently asked questions | `GROUP BY query` + `COUNT(*)`, ordered descending |
| Queries where no answer was found | `WHERE no_answer = 1` |
| Average response latency | `AVG(latency_ms)` |
| Total queries (extra) | `COUNT(*)` |

> **Note on the SQL part:** Since there is only one static document, the SQL component is about **usage analytics, not document content**. After the pipeline was working, 30+ test queries (a mix of answerable and out-of-scope) were run against `/ask` to populate the logs with realistic data.

---

## 8. Assumptions & Design Notes

- The system answers questions about a **single, static document** (the AWS Customer Agreement). `/ingest` processes that provided PDF from a known path   no document upload is required.
- The "most frequently asked" metric groups on the raw query string, so trivially different phrasings count separately. This is acceptable for the assignment's test-query workload.
- A free, locally-run embedding model is used so the system needs no paid embedding API. Only the generation step calls a hosted LLM (Groq free tier).
- Generated artifacts (FAISS index, chunks, SQLite DB) live in `storage/` and are gitignored   they are rebuilt by `/ingest`.

---

## 9. Demo

A short demo video/GIF (2–3 minutes) showing the app running asking a question, viewing the answer with sources, and the analytics dashboard   is available here:

**[Demo link: https://drive.google.com/file/d/1OKZUe-m9_URDRtgfbSKPbqKwCWqgXCEc/view?usp=sharing]**

---

## 10. Requirements

```
fastapi
uvicorn
streamlit
sentence-transformers
faiss-cpu
groq
python-dotenv
pymupdf
requests
```