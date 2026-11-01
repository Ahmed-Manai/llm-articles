"""Step 1 of RAG: load text files and split them into overlapping chunks."""
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class Document:
    """A piece of text plus the name of the file it came from (used for citations)."""
    text: str
    source: str


class TextFileLoader:
    """Loads one .txt file, or every .txt file inside a folder."""

    def __init__(self, path: str, encoding: str = "utf-8"):
        self.path = Path(path)
        self.encoding = encoding

    def load_documents(self) -> List[Document]:
        if self.path.is_dir():
            files = sorted(self.path.rglob("*.txt"))
        elif self.path.is_file() and self.path.suffix == ".txt":
            files = [self.path]
        else:
            raise ValueError(f"{self.path} is neither a folder nor a .txt file.")
        return [Document(f.read_text(encoding=self.encoding), f.name) for f in files]


class CharacterTextSplitter:
    """Cuts text into fixed-size chunks. Neighbouring chunks share `chunk_overlap`
    characters so a sentence on a boundary still appears whole in one of them."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        if chunk_size <= chunk_overlap:
            raise ValueError("chunk_size must be greater than chunk_overlap")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str) -> List[str]:
        step = self.chunk_size - self.chunk_overlap
        return [text[i : i + self.chunk_size] for i in range(0, len(text), step)]

    def split_documents(self, documents: List[Document]) -> List[Document]:
        return [
            Document(chunk, doc.source)
            for doc in documents
            for chunk in self.split(doc.text)
        ]


if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parent.parent / "data"
    docs = TextFileLoader(data_dir).load_documents()
    chunks = CharacterTextSplitter().split_documents(docs)
    print(f"{len(docs)} documents -> {len(chunks)} chunks")
    print(chunks[0].source, "|", chunks[0].text[:200])
