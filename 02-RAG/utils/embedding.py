"""Step 2 of RAG: turn text into embedding vectors with the Gemini API."""
import os
from getpass import getpass
from typing import List

import numpy as np
from dotenv import load_dotenv  # pip install python-dotenv
from google import genai  # pip install google-genai
from google.genai import types

# One embedding request accepts at most 100 texts, so larger lists are sent in batches.
BATCH_SIZE = 100


def get_client() -> genai.Client:
    """Create a Gemini client. The key comes from .env, the environment, or a prompt."""
    load_dotenv()  # looks for a .env file in this folder and its parents
    if not os.getenv("GEMINI_API_KEY"):
        os.environ["GEMINI_API_KEY"] = getpass("Gemini API key: ")
    # Retry automatically on rate limits (429) and temporary server errors.
    retry = types.HttpRetryOptions(attempts=5, initial_delay=2.0)
    return genai.Client(
        api_key=os.environ["GEMINI_API_KEY"],
        http_options=types.HttpOptions(retry_options=retry),
    )


class EmbeddingModel:
    def __init__(self, model_name: str = None, client: genai.Client = None):
        self.model_name = model_name or os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
        self.client = client or get_client()

    def get_embeddings(self, texts: List[str], task_type: str = "RETRIEVAL_DOCUMENT") -> np.ndarray:
        """Embed many texts. Returns an array of shape (len(texts), dimensions).

        task_type tells the model how the vectors will be used:
        RETRIEVAL_DOCUMENT for stored chunks, RETRIEVAL_QUERY for questions.
        """
        vectors = []
        for start in range(0, len(texts), BATCH_SIZE):
            response = self.client.models.embed_content(
                model=self.model_name,
                contents=texts[start : start + BATCH_SIZE],
                config=types.EmbedContentConfig(task_type=task_type),
            )
            vectors.extend(e.values for e in response.embeddings)
        return np.array(vectors)

    def get_embedding(self, text: str, task_type: str = "RETRIEVAL_QUERY") -> np.ndarray:
        """Embed a single text (by default, a search query)."""
        return self.get_embeddings([text], task_type=task_type)[0]


if __name__ == "__main__":
    model = EmbeddingModel()
    vectors = model.get_embeddings(["Hello, world!", "Goodbye, world!"])
    print(vectors.shape, vectors[0][:5])
