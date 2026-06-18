# Lightweight 384-dim sentence embedding model; fast enough for real-time retrieval
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Groq-hosted Llama 3.3 70B; low latency via Groq inference API
GENERATIVE_MODEL_NAME = "llama-3.3-70b-versatile"

# Target word count per chunk before the token-limit check trims it down
CHUNK_SIZE = 180

# Words shared between consecutive chunks to preserve sentence context at boundaries
OVERLAP_SIZE = 27

# Hard ceiling on tokens per chunk; _fit_chunk shrinks the chunk if this is exceeded
TOKEN_LIMIT = 200

# Low temperature keeps answers grounded and deterministic for a factual Q&A use case
TEMPERATURE = 0.1

INDEX_PATH = "../storage/index.faiss"
CHUNKS_PATH = "../storage/chunks.json"
DB_PATH = "../storage/query_logs.db"
DOCUMENT_PATH = "../data/AWS Customer Agreement.pdf"

# Cosine similarity floor below which retrieval is considered too weak to attempt generation
SCORE_THRESHOLD = 0.1

# Exact string the LLM returns when the answer is not in the context; used for no_answer detection
NO_ANSWER_PHRASE = "I could not find the answer in the document."



