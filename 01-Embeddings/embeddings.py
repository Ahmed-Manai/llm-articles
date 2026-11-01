"""Embeddings in practice with the Gemini API (code from embeddings.ipynb).

Run from anywhere: python 01-Embeddings/embeddings.py"""


# --- Setup: load the API key and create a Gemini client ---
import os
from getpass import getpass

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()  # reads GEMINI_API_KEY from a .env file in this folder or a parent folder
if not os.getenv("GEMINI_API_KEY"):
    os.environ["GEMINI_API_KEY"] = getpass("Gemini API key: ")

client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"],
    # retry automatically if the free tier rate limit (HTTP 429) is hit
    http_options=types.HttpOptions(retry_options=types.HttpRetryOptions(attempts=5, initial_delay=2.0)),
)
EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
print("Using", EMBEDDING_MODEL)


# --- 1. Cosine similarity ---
def cosine_similarity(a, b) -> float:
    """Cosine of the angle between two vectors: 1 = same direction, 0 = unrelated, -1 = opposite."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


a = [1, 2, 3]
print("same direction, twice as long:", round(cosine_similarity(a, [2, 4, 6]), 4))
print("at right angles:              ", round(cosine_similarity(a, [3, 0, -1]), 4))
print("opposite direction:           ", round(cosine_similarity(a, [-1, -2, -3]), 4))
print("partly aligned:               ", round(cosine_similarity(a, [3, 2, 1]), 4))


# --- 2. An embedding helper ---
def embed(texts, task_type="SEMANTIC_SIMILARITY") -> np.ndarray:
    """Return one embedding per text, as an array of shape (len(texts), dimensions)."""
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=list(texts),
        config=types.EmbedContentConfig(task_type=task_type),
    )
    return np.array([e.values for e in response.embeddings])


vector = embed(["I love puppies!"])[0]
print("dimensions:  ", vector.shape[0])
print("first values:", np.round(vector[:6], 4))
print("length (L2): ", round(float(np.linalg.norm(vector)), 4))


# --- 3. Comparing sentences ---
sentences = [
    "I love puppies!",       # 0  reference
    "I love dogs!",          # 1  similar meaning
    "I dislike cats!",       # 2  different feeling AND different animal
    "I love cats!",          # 3  only the animal changes
    "I dislike puppies!",    # 4  only the feeling changes
    "The quarterly tax report is due on Friday.",  # 5  unrelated topic
]
vectors = embed(sentences)

pairs = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5)]
for i, j in pairs:
    score = cosine_similarity(vectors[i], vectors[j])
    print(f"{score:.3f}   {sentences[i]!r}  vs  {sentences[j]!r}")


# --- 4. Vector arithmetic with a baseline ---
words = ["king", "man", "woman", "queen", "princess", "prince", "monarch", "daughter", "banana"]
word_vectors = dict(zip(words, embed(words)))

result = word_vectors["king"] - word_vectors["man"] + word_vectors["woman"]

print(f"result vs queen : {cosine_similarity(result, word_vectors['queen']):.3f}")
print(f"king   vs queen : {cosine_similarity(word_vectors['king'], word_vectors['queen']):.3f}   <- baseline")

print("\nNearest candidates (input words excluded):")
candidates = [w for w in words if w not in ("king", "man", "woman")]
ranking = sorted(candidates, key=lambda w: cosine_similarity(result, word_vectors[w]), reverse=True)
for w in ranking:
    print(f"  {cosine_similarity(result, word_vectors[w]):.3f}  {w}")


# --- 5. A tiny semantic search engine ---
documents = [
    "Puppies need several small meals a day while they grow.",
    "Adult dogs are usually fed once or twice daily.",
    "Kittens sleep up to 20 hours a day.",
    "A carburetor mixes air and fuel for an internal combustion engine.",
    "The queen is the most powerful piece in chess.",
]
doc_vectors = embed(documents, task_type="RETRIEVAL_DOCUMENT")


def search(query: str, k: int = 3):
    query_vector = embed([query], task_type="RETRIEVAL_QUERY")[0]
    scores = [cosine_similarity(query_vector, d) for d in doc_vectors]
    best = np.argsort(scores)[::-1][:k]
    return [(scores[i], documents[i]) for i in best]


for score, doc in search("How often should I feed my young dog?"):
    print(f"{score:.3f}  {doc}")


# --- 6. Smaller vectors ---
short = client.models.embed_content(
    model=EMBEDDING_MODEL,
    contents=["I love puppies!", "I love dogs!"],
    config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY", output_dimensionality=768),
)
short_vectors = np.array([e.values for e in short.embeddings])
print("shape:", short_vectors.shape)
print("length before normalizing:", np.round(np.linalg.norm(short_vectors, axis=1), 4))

short_vectors /= np.linalg.norm(short_vectors, axis=1, keepdims=True)
print("length after normalizing: ", np.round(np.linalg.norm(short_vectors, axis=1), 4))
print(f"similarity at 768 dims:  {short_vectors[0] @ short_vectors[1]:.3f}")
print(f"similarity at full size: {cosine_similarity(vectors[0], vectors[1]):.3f}")
