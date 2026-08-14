from typing import List, Tuple
import numpy as np


def normalize_vector(v: np.ndarray) -> np.ndarray:
    """Computes the L2 unit norm of a 1D vector."""
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def dot_product(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Calculates algebraic dot product between two 1D vectors."""
    if vec_a.shape != vec_b.shape:
        raise ValueError(
            f"Dimension mismatch: {vec_a.shape} != {vec_b.shape}"
        )
    return float(np.dot(vec_a, vec_b))


def euclidean_distance(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Calculates Euclidean (L2) distance between two 1D vectors."""
    if vec_a.shape != vec_b.shape:
        raise ValueError(
            f"Dimension mismatch: {vec_a.shape} != {vec_b.shape}"
        )
    return float(np.linalg.norm(vec_a - vec_b))


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Calculates Cosine Similarity: (A . B) / (||A|| * ||B||)."""
    if vec_a.shape != vec_b.shape:
        raise ValueError(
            f"Dimension mismatch: {vec_a.shape} != {vec_b.shape}"
        )

    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def batch_cosine_similarity(
    query_vector: np.ndarray, matrix: np.ndarray
) -> np.ndarray:
    """
    Computes cosine similarity of a 1D query vector against a 2D matrix of shape (N, D).
    Uses vectorized broadcasting for O(N) evaluation.
    """
    if query_vector.ndim != 1 or matrix.ndim != 2:
        raise ValueError("query_vector must be 1D and matrix must be 2D.")
    if query_vector.shape[0] != matrix.shape[1]:
        raise ValueError(
            f"Dimension mismatch: query dim {query_vector.shape[0]} != matrix dim {matrix.shape[1]}"
        )

    query_norm = np.linalg.norm(query_vector)
    matrix_norms = np.linalg.norm(matrix, axis=1)

    # Avoid division by zero
    query_norm = 1.0 if query_norm == 0 else query_norm
    matrix_norms = np.where(matrix_norms == 0, 1.0, matrix_norms)

    # Vectorized dot products: (N, D) @ (D,) -> (N,)
    dot_products = matrix @ query_vector
    similarities = dot_products / (matrix_norms * query_norm)

    return similarities