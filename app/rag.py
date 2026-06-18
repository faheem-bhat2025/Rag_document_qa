import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import re
import fitz
import faiss
from typing import List
from dotenv import load_dotenv
from groq import Groq
load_dotenv()       
from sentence_transformers import SentenceTransformer
from config import *
import json



class Rag:
    def __init__(self):
        self.document_path = DOCUMENT_PATH
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.index = None
        self.chunks = None

    def extract_text(self) -> str:
        doc = fitz.open(self.document_path)
        text = "".join(page.get_text() for page in doc)
        doc.close()
        return text

    def preprocess_text(self, text: str) -> str:
        text = text.replace('\xa0', ' ').replace('-\n', '').replace('\n', ' ')
        text = re.sub(r'\s+', ' ', text).replace("*", '')
        return text.strip()

    def _fit_chunk(self, words, start, end):
        end = min(end, len(words))
        sent = " ".join(words[start:end])
        ttl = len(self.model.tokenizer.encode(sent))
        if ttl <= TOKEN_LIMIT:
            return {"start": start, "end": end, "text": sent, "token_length": ttl}
        return self._fit_chunk(words, start, end - 10)

    def chunker(self, text: str) -> List[str]:
        words = text.split(" ")
        chunks, start = [], 0
        while start < len(words):
            value = self._fit_chunk(words, start, start + CHUNK_SIZE)
            chunks.append(value["text"])
            if value["end"] >= len(words):
                break
            start = value["end"] - OVERLAP_SIZE
        return chunks

    def ingest(self):
        text = self.preprocess_text(self.extract_text())
        self.chunks = self.chunker(text)
        embeddings = self.model.encode(
            self.chunks, convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        return len(self.chunks)

    def retrieve(self, query, k=4):
        q_emb = self.model.encode([query], convert_to_numpy=True,
                                  normalize_embeddings=True).astype("float32")
        scores, indices = self.index.search(q_emb, k)
        return [{"text": self.chunks[i], "score": float(s)}
                for s, i in zip(scores[0], indices[0])]

    def generate_answer(self, query, retrieved_chunks):
        """Build a grounded prompt from retrieved chunks and call the LLM."""
        context = "\n\n---\n\n".join(c["text"] for c in retrieved_chunks)
        system_prompt = (
            "You are a question-answering assistant for the AWS Customer Agreement. "
            "Answer the user's question using ONLY the provided context. "
            "If the answer is not contained in the context, reply exactly: "
            "'I could not find the answer in the document.' "
            "Do not use outside knowledge. Do not guess."
        )
        user_prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        resp = self.client.chat.completions.create(
            model=GENERATIVE_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
        )
        return resp.choices[0].message.content
    

    def save(self, index_path=INDEX_PATH, chunks_path=CHUNKS_PATH):
        """Persist the FAISS index and chunks to disk after ingestion."""
        faiss.write_index(self.index, index_path)
        with open(chunks_path, "w") as f:
            json.dump(self.chunks, f)

    def load(self, index_path=INDEX_PATH, chunks_path=CHUNKS_PATH):
        """Load a previously-built index and chunks from disk."""
        self.index = faiss.read_index(index_path)
        with open(chunks_path) as f:
            self.chunks = json.load(f)

    def is_ready(self):
        """True if an index is loaded and ready to serve queries."""
        return self.index is not None and self.chunks is not None