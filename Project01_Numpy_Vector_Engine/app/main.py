from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np

from .vector_math import (
    cosine_similarity,
    euclidean_distance,
    dot_product,
    normalize_vector,
    batch_cosine_similarity,
)

app = FastAPI(
    title="NumPy Vector Engine API",
    description="Microservice for raw linear algebra and vector similarity metrics.",
    version="1.0.0",
)

# Enable CORS for React frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request / Response Schemas ---
class PairVectorRequest(BaseModel):
    vector_a: List[float] = Field(..., example=[1.0, 2.0, 3.0])
    vector_b: List[float] = Field(..., example=[4.0, 5.0, 6.0])


class SingleVectorRequest(BaseModel):
    vector: List[float] = Field(..., example=[3.0, 4.0])


class BatchSearchRequest(BaseModel):
    query_vector: List[float] = Field(..., example=[1.0, 0.0, 0.0])
    candidate_vectors: List[List[float]] = Field(
        ...,
        example=[
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.707, 0.707, 0.0],
        ],
    )
    top_k: int = Field(default=3, ge=1)


# --- API Routes ---
@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "NumPy Vector Engine"}


@app.post("/api/vectors/compare")
def compare_vectors(payload: PairVectorRequest):
    """Calculates Cosine Similarity, Euclidean Distance, and Dot Product between two vectors."""
    try:
        a = np.array(payload.vector_a, dtype=np.float32)
        b = np.array(payload.vector_b, dtype=np.float32)

        return {
            "cosine_similarity": round(cosine_similarity(a, b), 5),
            "euclidean_distance": round(euclidean_distance(a, b), 5),
            "dot_product": round(dot_product(a, b), 5),
            "dimension": int(a.shape[0]),
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/vectors/normalize")
def normalize(payload: SingleVectorRequest):
    """Computes the L2 normalized unit vector."""
    v = np.array(payload.vector, dtype=np.float32)
    normalized = normalize_vector(v)
    return {
        "original_vector": payload.vector,
        "normalized_vector": [round(float(x), 5) for x in normalized],
        "original_norm": round(float(np.linalg.norm(v)), 5),
    }


@app.post("/api/vectors/batch-search")
def batch_search(payload: BatchSearchRequest):
    """Ranks candidate vectors by cosine similarity to a query vector."""
    try:
        query = np.array(payload.query_vector, dtype=np.float32)
        matrix = np.array(payload.candidate_vectors, dtype=np.float32)

        scores = batch_cosine_similarity(query, matrix)

        # Pair candidates with their original indices and scores
        results = [
            {
                "index": int(i),
                "vector": payload.candidate_vectors[i],
                "similarity_score": round(float(scores[i]), 5),
            }
            for i in range(len(scores))
        ]

        # Sort descending by similarity score
        results.sort(key=lambda x: x["similarity_score"], reverse=True)

        return {
            "top_results": results[: payload.top_k],
            "total_candidates": len(payload.candidate_vectors),
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))