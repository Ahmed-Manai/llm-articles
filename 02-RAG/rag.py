"""Retrieval-augmented generation from scratch with Gemini (code from rag.ipynb).

Run from anywhere: python 02-RAG/rag.py"""


# --- Setup ---
from pathlib import Path

from utils.chatmodel import ChatGemini
from utils.embedding import EmbeddingModel, get_client
from utils.prompts import SystemRolePrompt, UserRolePrompt
from utils.text_utils import CharacterTextSplitter, TextFileLoader
from utils.vectordatabase import VectorDatabase

DATA_DIR = Path("data") if Path("data").exists() else Path(__file__).resolve().parent / "data"
client = get_client()  # one client, shared by the embedding model and the chat model


# --- Step 1: load the documents ---
documents = TextFileLoader(DATA_DIR).load_documents()

print(f"Loaded {len(documents)} documents:")
for doc in documents:
    print(f"  {doc.source:32} {len(doc.text):>5} characters")


# --- Step 2: split into chunks ---
splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(documents)
print(f"{len(documents)} documents -> {len(chunks)} chunks\n")

first, second = chunks[0], chunks[1]
shared = first.text[-splitter.chunk_overlap:]
print("Last 200 characters of chunk 0 == first 200 of chunk 1?", second.text.startswith(shared))
print("Shared text:", repr(shared))


# --- Step 3: embed and store ---
embedding_model = EmbeddingModel(client=client)
vector_db = VectorDatabase(embedding_model).build_from_documents(chunks)
print("Vector matrix shape:", vector_db.vectors.shape, "(chunks x dimensions)")


# --- Step 4: retrieve ---
question = "What is the Two-Source Rule?"

for chunk, score in vector_db.search_by_text(question, k=3):
    contains = "yes" if "Two-Source Rule" in chunk.text else "no "
    preview = " ".join(chunk.text.split())[:70]
    print(f"{score:.3f}  [{chunk.source}]  defines the rule: {contains}  | starts: {preview}...")


# --- Step 5: the model on its own ---
chat_model = ChatGemini(client=client)
print("Chat model:", chat_model.model_name, "\n")

no_rag_answer = chat_model.run([
    {"role": "user", "content": "At Brightwave Labs, what is the Two-Source Rule?"}
])
print(no_rag_answer)


# --- Step 6: the prompt templates ---
RAG_SYSTEM_TEMPLATE = """You are Atlas, an internal assistant for Brightwave Labs.
Answer the question using ONLY the context provided.
After each fact, cite its source file in square brackets, for example [02-ai-usage-policy.txt].
If the context does not contain the answer, reply exactly: "I don't know based on the available documents."
Keep answers short and factual."""

RAG_USER_TEMPLATE = """Context:
{context}

Question:
{question}"""

system_prompt = SystemRolePrompt(RAG_SYSTEM_TEMPLATE)
user_prompt = UserRolePrompt(RAG_USER_TEMPLATE)
print("User prompt variables:", user_prompt.get_input_variables())


# --- Step 7: the RAG pipeline ---
class RAGPipeline:
    def __init__(self, llm: ChatGemini, vector_db: VectorDatabase, k: int = 4):
        self.llm = llm
        self.vector_db = vector_db
        self.k = k

    def run(self, question: str) -> dict:
        retrieved = self.vector_db.search_by_text(question, k=self.k)
        context = "\n\n".join(f"[{chunk.source}]\n{chunk.text}" for chunk, _ in retrieved)
        messages = [
            system_prompt.create_message(),
            user_prompt.create_message(context=context, question=question),
        ]
        return {"answer": self.llm.run(messages), "context": retrieved}


rag = RAGPipeline(chat_model, vector_db)
result = rag.run("What is the Two-Source Rule?")
print(result["answer"])


# --- Step 8: more questions ---
questions = [
    "What chunk size does Atlas use, and why was it chosen?",
    "Who has to approve a new AI model vendor, and in what order?",
    "How many vacation days do I get after four years?",
    "Who won the 2022 FIFA World Cup?",  # not in the documents
]

for q in questions:
    result = rag.run(q)
    sources = sorted({chunk.source for chunk, _ in result["context"]})
    print(f"Q: {q}\nA: {result['answer'].strip()}\n   retrieved from: {', '.join(sources)}\n")
