import numpy as np
import pytest
from Project01_Numpy_Vector_Engine.app.vector_math import (
    cosine_similarity,
    euclidean_distance,
    dot_product,
    normalize_vector,
    batch_cosine_similarity,
)


def test_identical_vectors_cosine():
    a = np.array([1.0, 2.0, 3.0])
    assert pytest.approx(cosine_similarity(a, a), 0.0001) == 1.0


def test_orthogonal_vectors_cosine():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert pytest.approx(cosine_similarity(a, b), 0.0001) == 0.0


def test_euclidean_distance():
    a = np.array([0.0, 0.0])
    b = np.array([3.0, 4.0])
    assert pytest.approx(euclidean_distance(a, b), 0.0001) == 5.0


def test_batch_cosine_similarity():
    query = np.array([1.0, 0.0])
    matrix = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    scores = batch_cosine_similarity(query, matrix)
    assert pytest.approx(scores[0], 0.0001) == 1.0
    assert pytest.approx(scores[1], 0.0001) == 0.0
    assert pytest.approx(scores[2], 0.0001) == -1.0