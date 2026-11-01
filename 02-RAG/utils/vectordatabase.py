"""Step 3 of RAG: store chunk vectors and find the ones closest to a question."""
from typing import List, Tuple

import numpy as np

from .embedding import EmbeddingModel
from .text_utils import Document


def cosine_similarity(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Cosine similarity between one vector and every row of a matrix."""
    return (matrix @ vector) / (np.linalg.norm(matrix, axis=1) * np.linalg.norm(vector))


class VectorDatabase:
    """A tiny in-memory vector store: one row per chunk, searched by brute force.
    Fine for thousands of chunks; real systems use an ANN index (FAISS, HNSW...)."""

    def __init__(self, embedding_model: EmbeddingModel = None):
        self.embedding_model = embedding_model or EmbeddingModel()
        self.chunks: List[Document] = []
        self.vectors = np.empty((0, 0))

    def build_from_documents(self, chunks: List[Document]) -> "VectorDatabase":
        self.chunks = list(chunks)
        self.vectors = self.embedding_model.get_embeddings(
            [c.text for c in self.chunks], task_type="RETRIEVAL_DOCUMENT"
        )
        return self

    def search(self, query_vector: np.ndarray, k: int = 4) -> List[Tuple[Document, float]]:
        scores = cosine_similarity(self.vectors, query_vector)
        best = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in best]

    def search_by_text(self, query: str, k: int = 4) -> List[Tuple[Document, float]]:
        query_vector = self.embedding_model.get_embedding(query, task_type="RETRIEVAL_QUERY")
        return self.search(query_vector, k)


if __name__ == "__main__":
    texts = [
        "I like to eat broccoli and bananas.",
        "I ate a banana and spinach smoothie for breakfast.",
        "Chinchillas and kittens are cute.",
        "My sister adopted a kitten yesterday.",
    ]
    db = VectorDatabase().build_from_documents([Document(t, "example") for t in texts])
    for chunk, score in db.search_by_text("I think fruit is awesome!", k=2):
        print(f"{score:.3f}  {chunk.text}")
